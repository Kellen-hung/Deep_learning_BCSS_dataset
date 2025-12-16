

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using Device: {DEVICE}")

# ĐƯỜNG DẪN
TRAIN_IMAGE_PATH = '/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS/train'
VAL_IMAGE_PATH   = '/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS/val'
TRAIN_MASK_PATH  = '/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS/train_mask'
VAL_MASK_PATH    = '/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS/val_mask'

NUM_CLASSES = 3  
IMG_SIZE = 224
BATCH_SIZE = 16 
EPOCHS = 20      
LEARNING_RATE = 6e-5 

# ===================== DATASET =====================
class BCSSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        if mask.ndim == 3: mask = mask[:, :, 0]
        mask[mask >= NUM_CLASSES] = 0 

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask'].long()
            
        return image, mask

# ===================== AUGMENTATION =====================
transforms_train = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.OneOf([
        A.GridDistortion(p=1.0),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
    ], p=0.3),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

transforms_val = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

train_ds = BCSSDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, transforms_train)
val_ds   = BCSSDataset(VAL_IMAGE_PATH, VAL_MASK_PATH, transforms_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ===================== LOSS & METRICS (FIXED) =====================
ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)

def dice_loss_multiclass(logits, targets, smooth=1e-6):
    """Hàm tính Loss dựa trên Dice - ĐÃ FIX LỖI DIM"""
    probs = F.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
    
    intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
    cardinality = torch.sum(probs + targets_one_hot, dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()

def criterion(outputs, masks):
    logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    return 0.5 * ce_loss(logits, masks) + 0.5 * dice_loss_multiclass(logits, masks)

def compute_metrics(preds, labels, num_classes):
    preds = preds.view(-1)
    labels = labels.view(-1)
    ious, dices = [], []
    
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        cardinality = pred_inds.sum().item() + target_inds.sum().item()
        
        if union == 0: ious.append(float('nan'))
        else: ious.append(intersection / union)
            
        if cardinality == 0: dices.append(float('nan'))
        else: dices.append((2. * intersection) / cardinality)
            
    return np.nanmean(ious), np.nanmean(dices)

# ===================== MODEL =====================
print("🛠️ Đang tải model nvidia/mit-b1...")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b1", 
    num_labels=NUM_CLASSES, 
    ignore_mismatched_sizes=True
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.15
)

# ===================== TRAINING LOOP =====================
best_dice = 0.0
history = {'train_loss': [], 'train_iou': [], 'train_dice': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}

for epoch in range(1, EPOCHS + 1):
    # --- TRAIN ---
    model.train()
    train_loss, train_iou, train_dice = 0.0, 0.0, 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
    for images, masks in pbar:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        
        with torch.no_grad():
            logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1)
            iou, dice = compute_metrics(preds, masks, NUM_CLASSES)
            train_iou += iou
            train_dice += dice
            
        pbar.set_postfix(loss=loss.item(), iou=iou, dice=dice)
        
    avg_train_loss = train_loss / len(train_loader)
    avg_train_iou  = train_iou / len(train_loader)
    avg_train_dice = train_dice / len(train_loader)

    # --- VALID ---
    model.eval()
    val_loss, val_iou, val_dice = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(pixel_values=images)
            
            logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            v_loss = criterion(outputs, masks)
            val_loss += v_loss.item()
            
            preds = torch.argmax(logits, dim=1)
            iou, dice = compute_metrics(preds, masks, NUM_CLASSES)
            val_iou += iou
            val_dice += dice
            
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou  = val_iou / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    
    history['train_loss'].append(avg_train_loss)
    history['train_dice'].append(avg_train_dice)
    history['val_loss'].append(avg_val_loss)
    history['val_dice'].append(avg_val_dice)
    
    print(f"DONE Ep {epoch} | T.Loss: {avg_train_loss:.4f} | T.Dice: {avg_train_dice:.4f} || V.Loss: {avg_val_loss:.4f} | V.IoU: {avg_val_iou:.4f} | V.Dice: {avg_val_dice:.4f}")

    if avg_val_dice > best_dice:
        print(f"New Best Dice: {avg_val_dice:.4f} --> Saving...")
        best_dice = avg_val_dice
        torch.save(model.state_dict(), "best_segformer_b1_full.pth")

# Vẽ biểu đồ
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_dice'], label='Train Dice')
plt.plot(history['val_dice'], label='Val Dice')
plt.title('Dice Score History')
plt.legend()
plt.savefig('training_result_full.png')
print(f"\n✅ TRAINING DONE! Best Val Dice: {best_dice:.4f}")
