import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
from transformers import SegformerForSemanticSegmentation
import torch.cuda.amp as amp

# ==========================================
# 1. CONFIGURATION (CẤU HÌNH "BẤT TỬ")
# ==========================================
BASE_DIR = "/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS_512"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_512")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "train_mask_512")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val_512")
VAL_MASK_DIR = os.path.join(BASE_DIR, "val_mask_512")

NUM_CLASSES = 24 
IMG_SIZE = 512
NUM_EPOCHS = 30
BATCH_SIZE = 6 
ACCUMULATION_STEPS = 4 
NUM_WORKERS = 2
LEARNING_RATE = 6e-5 
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"⚙️ Device: {DEVICE}")
print(f"🛡️ Strategy: Full Spectrum Training ({NUM_CLASSES} slots)")

# ==========================================
# 2. DATASET (INPUT SANITIZATION)
# ==========================================
class BCSSDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        
        if mask.ndim == 3: mask = mask[:, :, 0]

        mask[mask >= NUM_CLASSES] = 0 
        
        mask = mask.astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()
        else:
            image = torch.from_numpy(image).float().permute(2,0,1)/255.0
            mask = torch.from_numpy(mask).long()
            
        return image, mask

# ==========================================
# 3. AUGMENTATION
# ==========================================
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE), 
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.OneOf([
        A.GridDistortion(p=1.0),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
    ], p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

train_dataset = BCSSDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
val_dataset = BCSSDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ==========================================
# 4. LOSS & OPTIMIZER (IMBALANCE HANDLING)
# ==========================================
def get_fixed_weights(device):
    weights = torch.ones(NUM_CLASSES)
    weights[0] = 0.5; weights[1] = 0.5; weights[2] = 0.6; weights[3] = 0.8
    rare_indices = [5, 6, 9, 10, 13, 18] 
    for idx in rare_indices:
        if idx < NUM_CLASSES: weights[idx] = 20.0 
    if 6 < NUM_CLASSES: weights[6] = 50.0
    return weights.to(device)

class_weights = get_fixed_weights(DEVICE)
print("Class Weights Applied.")

ce_loss = nn.CrossEntropyLoss(weight=class_weights)

def safe_dice_loss(logits, targets, smooth=1e-6):
    probs = F.softmax(logits, dim=1)
    
    targets_safe = targets.clone()
    targets_safe[targets_safe >= logits.shape[1]] = 0 
    
    targets_one_hot = F.one_hot(targets_safe, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    
    dims = (2, 3)
    intersection = torch.sum(probs * targets_one_hot, dims)
    cardinality = torch.sum(probs + targets_one_hot, dims)
    
    dice = (2. * intersection + smooth) / (cardinality + smooth)
    
    return 1.0 - dice.mean()

def criterion(outputs, masks):
    logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    
    return 0.5 * ce_loss(logits, masks) + 0.5 * safe_dice_loss(logits, masks)

# ==========================================
# 5. METRICS (SMART SCORE)
# ==========================================
def compute_smart_metrics(preds, masks):
    preds = preds.view(-1)
    masks = masks.view(-1)
    present_classes = torch.unique(masks)
    
    dice_list = []
    iou_list = []
    
    for cls in present_classes:
        cls = cls.item()
        if cls == 0: continue 
        
        p = (preds == cls)
        t = (masks == cls)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        
        if union > 0:
            iou = inter / union
            dice = (2 * inter) / (p.sum().item() + t.sum().item() + 1e-6)
            iou_list.append(iou)
            dice_list.append(dice)
            
    return np.mean(dice_list) if dice_list else 0, np.mean(iou_list) if iou_list else 0

# ==========================================
# 6. MODEL SETUP
# ==========================================
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b1", 
    num_labels=NUM_CLASSES, 
    ignore_mismatched_sizes=True
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader)//ACCUMULATION_STEPS, epochs=NUM_EPOCHS, pct_start=0.3
)
scaler = amp.GradScaler()

# ==========================================
# 7. TRAINING LOOP
# ==========================================
history = {'train_loss': [], 'val_dice': [], 'val_iou': []}
best_dice = 0.0

print("\nSTART TRAINING (Full Spectrum 24 Classes)...")

for epoch in range(NUM_EPOCHS):
    # --- TRAIN ---
    model.train()
    train_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
    for i, (images, masks) in enumerate(pbar):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        with amp.autocast():
            outputs = model(pixel_values=images)
            loss = criterion(outputs, masks)
            loss = loss / ACCUMULATION_STEPS
            
        scaler.scale(loss).backward()
        
        if (i+1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
        train_loss += loss.item() * ACCUMULATION_STEPS
        pbar.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)
        
    avg_train_loss = train_loss / len(train_loader)
    
    # --- VALIDATION ---
    model.eval()
    val_dice_acc = 0
    val_iou_acc = 0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="[Val]", leave=False):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            with amp.autocast():
                outputs = model(pixel_values=images)
                logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            
            preds = torch.argmax(logits, dim=1)
            
            # Tính smart metrics
            d, iou = compute_smart_metrics(preds, masks)
            val_dice_acc += d
            val_iou_acc += iou
            
    avg_val_dice = val_dice_acc / len(val_loader)
    avg_val_iou = val_iou_acc / len(val_loader)
    
    # --- LOGGING ---
    print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | Val IoU: {avg_val_iou:.4f}")
    
    history['train_loss'].append(avg_train_loss)
    history['val_dice'].append(avg_val_dice)
    history['val_iou'].append(avg_val_iou)
    
    # Save Best
    if avg_val_dice > best_dice:
        print(f" New Best Dice: {avg_val_dice:.4f} (IoU: {avg_val_iou:.4f}) -> Saving...")
        best_dice = avg_val_dice
        torch.save(model.state_dict(), "best_segformer_bcss_24cls.pth")

# ==========================================
# 8. VISUALIZATION REPORT
# ==========================================
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.title('Loss History')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_dice'], label='Val Dice')
plt.plot(history['val_iou'], label='Val IoU')
plt.title('Validation Metrics')
plt.legend()

plt.savefig('training_report.png')
pd.DataFrame(history).to_csv('training_history.csv', index=False)
print("\nTRAINING COMPLETED! Saved model as 'best_segformer_bcss_24cls.pth'")
