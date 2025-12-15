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

BASE_DIR = "/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS_512"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_512")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "train_mask_512")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val_512")
VAL_MASK_DIR = os.path.join(BASE_DIR, "val_mask_512")

NUM_CLASSES = 22
IMG_SIZE = 512
NUM_EPOCHS = 30
BATCH_SIZE = 6
ACCUMULATION_STEPS = 4
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.005

print(f"Device: {DEVICE}")


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

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()
        else:
            image = torch.from_numpy(image).float().permute(2,0,1)/255.0
            mask = torch.from_numpy(mask).long()
        return image, mask

# -------------------------
# AUGMENTATION
# -------------------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=25, p=0.3),
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
        A.ElasticTransform(alpha=1, sigma=30, alpha_affine=30, p=1.0)
    ], p=0.2),
    A.CoarseDropout(max_holes=4, max_height=16, max_width=16, min_holes=1, p=0.3),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.2),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

# -------------------------
# DATALOADER
# -------------------------
train_dataset = BCSSDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
val_dataset = BCSSDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# -------------------------
# CLASS WEIGHTS
# -------------------------
def compute_class_weights(loader, num_classes, num_batches=30):
    count = {i:0 for i in range(num_classes)}
    total = 0
    for i, (_, masks) in enumerate(loader):
        if i >= num_batches:
            break
        flat = masks.view(-1).numpy()
        uniques, counts = np.unique(flat, return_counts=True)
        for u, c in zip(uniques, counts):
            if u < num_classes:
                count[u] += c
        total += masks.numel()
    weights = []
    for i in range(num_classes):
        w = np.log(1.05 + total/count[i]) if count[i] > 0 else 1.0
        weights.append(w)
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()
    return weights.to(DEVICE)

class_weights = compute_class_weights(train_loader, NUM_CLASSES)

# -------------------------
# EARLY STOPPING
# -------------------------
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        score = metric
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f" ⏳ EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

early_stopping = EarlyStopping(patience=7)

# -------------------------
# MODEL, LOSS, OPTIMIZER
# -------------------------
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b1",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(DEVICE)

ce_loss = nn.CrossEntropyLoss(weight=class_weights)

def dice_loss_fn(preds, masks, eps=1e-6):
    num_classes = preds.shape[1]
    preds_soft = F.softmax(preds, dim=1)
    masks_one_hot = F.one_hot(masks, num_classes).permute(0,3,1,2).float()
    inter = (preds_soft * masks_one_hot).sum(dim=(2,3))
    union = preds_soft.sum(dim=(2,3)) + masks_one_hot.sum(dim=(2,3))
    dice = 1 - ((2 * inter + eps) / (union + eps)).mean()
    return dice

def criterion(outputs, masks):
    logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    return 0.2 * ce_loss(logits, masks) + 0.8 * dice_loss_fn(logits, masks)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    steps_per_epoch=len(train_loader)//ACCUMULATION_STEPS,
    epochs=NUM_EPOCHS,
    pct_start=0.4,
    anneal_strategy='cos'
)

scaler = amp.GradScaler()

# -------------------------
# HELPER: Dice Score
# -------------------------
def compute_dice_score(preds, masks, num_classes):
    dice_scores = []
    for c in range(1, num_classes):
        p = (preds == c)
        t = (masks == c)
        inter = (p & t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            continue
        dice_scores.append((2. * inter / (union + 1e-6)).item())
    return np.mean(dice_scores) if dice_scores else 0.0

# -------------------------
# TRAINING LOOP
# -------------------------
history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
best_dice = 0.0

print("\n START TRAINING...")

for epoch in range(NUM_EPOCHS):
    # ----- TRAIN -----
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    for i, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        with amp.autocast():
            outputs = model(pixel_values=images)
            loss = criterion(outputs, masks) / ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        train_loss += loss.item() * ACCUMULATION_STEPS
    avg_train_loss = train_loss / len(train_loader)

    # ----- VALIDATION -----
    model.eval()
    val_loss, val_dice = 0.0, 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            with amp.autocast():
                outputs = model(pixel_values=images)
                loss_val = criterion(outputs, masks)
            val_loss += loss_val.item()
            logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1)
            val_dice += compute_dice_score(preds, masks, NUM_CLASSES)
    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_dice'].append(avg_val_dice)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

    # ----- SAVE BEST MODEL -----
    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        model.save_pretrained("./best_segformer_bcss_512")
        print(f"Saved Best Model (Dice: {best_dice:.4f})")

    # ----- EARLY STOPPING -----
    early_stopping(avg_val_dice)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

    print("-"*50)

# -------------------------
# PLOT & SAVE HISTORY
# -------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss Curve')
plt.subplot(1,2,2)
plt.plot(history['val_dice'], label='Val Dice Score')
plt.legend(); plt.title('Dice Score')
plt.savefig('./training_results.png', dpi=300)

pd.DataFrame(history).to_csv('./training_history.csv', index=False)
print("\nDONE! Results saved.")
