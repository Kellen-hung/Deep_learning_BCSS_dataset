# confirm the format of dataset, using 512*512 pixels dataset, 21 classes
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

# 視覺化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base_dir = "/mnt/storage/kellen/deep_learning/BCSS"

for subdir in ["train", "train_mask", "val", "val_mask"]:
	path = os.path.join(base_dir, subdir)
	print(f"{subdir}: {len(os.listdir(path))} files")

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

		# read image and mask
		image = np.array(Image.open(img_path).convert("RGB"))
		mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

		if self.transform is not None:
			augmented = self.transform(image=image, mask=mask)
			image = augmented['image']
			mask = augmented['mask']
			mask = mask.long()
		else:
			# convert to tensor
			image = torch.from_numpy(image).permute(0, 1, 2).float()  # HWC → HWC, still fine
			image = image.permute(2, 0, 1)  # HWC → CHW
			mask = torch.from_numpy(mask).long()   # segmentation 必須 long 型別

		return image, mask
	
transforms_train = A.Compose([
	A.HorizontalFlip(p=0.5),
	A.VerticalFlip(p=0.5),
	A.RandomRotate90(p=0.5),
	A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
	A.GaussianBlur(blur_limit=(3, 7), p=0.5),
	A.ElasticTransform(alpha=1, sigma=50, p=0.5),
	A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	ToTensorV2()
])

transforms_val = A.Compose([
	A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	ToTensorV2()
])

train_dataset = BCSSDataset(
	img_dir=os.path.join(base_dir, "train"),
	mask_dir=os.path.join(base_dir, "train_mask"),
	transform=transforms_train
)

val_dataset = BCSSDataset(
	img_dir=os.path.join(base_dir, "val"),
	mask_dir=os.path.join(base_dir, "val_mask"),
	transform=transforms_val
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)

# # --- Basic building block ---
# class DoubleConv(nn.Module):
# 	def __init__(self, in_channels, out_channels):
# 		super().__init__()
# 		self.conv = nn.Sequential(
# 			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
# 			nn.BatchNorm2d(out_channels),
# 			nn.ReLU(inplace=True),
# 			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
# 			nn.BatchNorm2d(out_channels),
# 			nn.ReLU(inplace=True),
# 		)

# 	def forward(self, x):
# 		return self.conv(x)

# # --- Full UNet ---
# class UNet(nn.Module):
# 	def __init__(self, in_channels=3, out_channels=21):
# 		super().__init__()
# 		
# 		# Encoder
# 		self.down1 = DoubleConv(in_channels, 64)
# 		self.down2 = DoubleConv(64, 128)
# 		self.down3 = DoubleConv(128, 256)
# 		self.down4 = DoubleConv(256, 512)
# 		self.down5 = DoubleConv(512, 1024)

# 		# Decoder
# 		self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
# 		self.conv1 = DoubleConv(1024, 512)
# 		
# 		self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
# 		self.conv2 = DoubleConv(512, 256)

# 		self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
# 		self.conv3 = DoubleConv(256, 128)

# 		self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
# 		self.conv4 = DoubleConv(128, 64)

# 		self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

# 		self.pool = nn.MaxPool2d(2)

# 	def forward(self, x):
# 		# Encoder
# 		x1 = self.down1(x)
# 		x2 = self.down2(self.pool(x1))
# 		x3 = self.down3(self.pool(x2))
# 		x4 = self.down4(self.pool(x3))
# 		x5 = self.down5(self.pool(x4))

# 		# Decoder
# 		x = self.up1(x5)
# 		x = torch.cat([x, x4], dim=1)
# 		x = self.conv1(x)
# 		
# 		x = self.up2(x)
# 		x = torch.cat([x, x3], dim=1)
# 		x = self.conv2(x)

# 		x = self.up3(x)
# 		x = torch.cat([x, x2], dim=1)
# 		x = self.conv3(x)

# 		x = self.up4(x)
# 		x = torch.cat([x, x1], dim=1)
# 		x = self.conv4(x)

# 		return self.out_conv(x)

class DiceBCELoss(nn.Module):
	def __init__(self, weight=None, size_average=True):
		super(DiceBCELoss, self).__init__()
		# CrossEntropy 會自動處理多類別
		self.ce = nn.CrossEntropyLoss(weight=weight)

	def forward(self, inputs, targets, smooth=1):
		# inputs: (Batch, 22, 224, 224)
		# targets: (Batch, 224, 224)
		
		# 1. 先算標準 CrossEntropy Loss
		ce_loss = self.ce(inputs, targets)
		
		# 2. 接著算 Dice Loss
		inputs_prob = F.softmax(inputs, dim=1) # 轉成機率
		
		# 自動將 target 轉成 (Batch, 22, 224, 224) 的 one-hot
		# num_classes 會自動讀取 inputs 的 channel 數 (也就是 22)
		targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
		
		# 計算 Dice
		intersection = (inputs_prob * targets_one_hot).sum(dim=(2, 3))
		union = inputs_prob.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
		
		dice = (2. * intersection + smooth) / (union + smooth)
		
		# Dice Loss = 1 - Mean Dice Score
		dice_loss = 1 - dice.mean()
		
		# 3. 結合兩者 (可以嘗試調整權重，例如 0.5/0.5 或 0.7/0.3)
		return 0.5 * ce_loss + 0.5 * dice_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# model = UNet(in_channels=3, out_channels=21)

model = smp.Unet(
	encoder_name="resnet34",        # 骨幹網路 (Backbone)
	encoder_weights="imagenet",     # 載入 ImageNet 預訓練權重 (這就是外掛本體!)
	in_channels=3,                  # 輸入 RGB 圖片
	classes=21,                     # 輸出 21 個類別
)

if torch.cuda.device_count() > 1:
	print("Using", torch.cuda.device_count(), "GPUs!")
	model = nn.DataParallel(model)

model = model.to(device)

# criterion = nn.CrossEntropyLoss()
criterion = DiceBCELoss()

# 1. 定義組合 Loss：Focal Loss (專注難題) + Dice Loss (專注形狀)
# mode='multiclass' 會自動處理 21 個類別的 softmax 和 one-hot
# dice_loss = smp.losses.DiceLoss(mode='multiclass')
# focal_loss = smp.losses.FocalLoss(mode='multiclass')

# # 2. 定義新的 Loss Function
# def criterion(preds, targets):
# 	return dice_loss(preds, targets) + focal_loss(preds, targets)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

num_epochs = 50

def pixel_accuracy(pred, mask):
	return (pred == mask).float().mean().item()

def dice_coefficient(pred, mask, num_classes=21, eps=1e-6):
	dice = 0.0
	for c in range(num_classes):
		pred_c = (pred == c).float()
		mask_c = (mask == c).float()
		intersection = (pred_c * mask_c).sum()
		union = pred_c.sum() + mask_c.sum()
		dice += (2 * intersection + eps) / (union + eps)
	return (dice / num_classes).item()

# 🆕 新增 IoU 計算
def iou_score(pred, mask, num_classes=21, eps=1e-6):
	"""計算 IoU (Intersection over Union)"""
	iou = 0.0
	for c in range(num_classes):
		pred_c = (pred == c).float()
		mask_c = (mask == c).float()
		intersection = (pred_c * mask_c).sum()
		union = pred_c.sum() + mask_c.sum() - intersection
		if union > 0:
			iou += (intersection + eps) / (union + eps)
		else:
			iou += 1.0  # 如果該類別不存在，視為完美預測
	return (iou / num_classes).item()

# ===== 🆕 訓練歷史記錄 =====
history = {
	'train_loss': [],
	'val_loss': [],
	'train_dice': [],
	'val_dice': [],
	'train_iou': [],
	'val_iou': [],
	'pixel_acc': []
}

best_dice = 0.0
best_epoch = 0

for epoch in range(num_epochs):
	# --- Training ---
	model.train()
	running_loss = 0.0
	train_dice_list = []  # 🆕 記錄訓練 dice
	train_iou_list = []   # 🆕 記錄訓練 iou
	
	for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
		images, masks = images.to(device), masks.to(device).long()
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, masks)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		
		# 🆕 計算訓練時的 dice 和 iou
		preds = torch.argmax(outputs, dim=1)
		train_dice_list.append(dice_coefficient(preds, masks))
		train_iou_list.append(iou_score(preds, masks))
	
	avg_train_loss = running_loss / len(train_loader)
	avg_train_dice = sum(train_dice_list) / len(train_dice_list)
	avg_train_iou = sum(train_iou_list) / len(train_iou_list)
	
	print(f"\nEpoch [{epoch+1}/{num_epochs}]")
	print(f"  Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, Train IoU: {avg_train_iou:.4f}")

	# --- Validation ---
	model.eval()
	val_loss = 0.0
	pixel_acc_list = []
	dice_list = []
	iou_list = []  # 🆕
	
	with torch.no_grad():
		for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
			images, masks = images.to(device), masks.to(device).long()
			outputs = model(images)
			preds = torch.argmax(outputs, dim=1)

			val_loss += criterion(outputs, masks).item()
			pixel_acc_list.append(pixel_accuracy(preds, masks))
			dice_list.append(dice_coefficient(preds, masks))
			iou_list.append(iou_score(preds, masks))  # 🆕
	
	avg_val_loss = val_loss / len(val_loader)
	avg_pixel_acc = sum(pixel_acc_list) / len(pixel_acc_list)
	avg_dice = sum(dice_list) / len(dice_list)
	avg_iou = sum(iou_list) / len(iou_list)  # 🆕

	print(f"  Val Loss: {avg_val_loss:.4f}, Pixel Acc: {avg_pixel_acc:.4f}, Val Dice: {avg_dice:.4f}, Val IoU: {avg_iou:.4f}")
	
	# ===== 🆕 記錄歷史 =====
	history['train_loss'].append(avg_train_loss)
	history['val_loss'].append(avg_val_loss)
	history['train_dice'].append(avg_train_dice)
	history['val_dice'].append(avg_dice)
	history['train_iou'].append(avg_train_iou)
	history['val_iou'].append(avg_iou)
	history['pixel_acc'].append(avg_pixel_acc)
	
	# ===== 🆕 儲存最佳模型 =====
	if avg_dice > best_dice:
		best_dice = avg_dice
		best_epoch = epoch + 1
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'val_dice': avg_dice,
		}, './best_model.pth')
		print(f"  ✓ Saved best model! Best Dice: {best_dice:.4f}")
	
	print("-" * 60)
	scheduler.step(avg_dice)
	
	with torch.no_grad():
		# 拿出 image, mask, pred
		debug_img = images[0].cpu().permute(1, 2, 0).numpy() # 轉回 HWC
		# 反正規化 (如果有的話，這裡簡化處理直接 clip)
		debug_img = (debug_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
		debug_img = np.clip(debug_img, 0, 1)
		
		debug_mask = masks[0].cpu().numpy()
		debug_pred = preds[0].cpu().numpy()

		# 畫圖
		fig, ax = plt.subplots(1, 3, figsize=(15, 5))
		ax[0].imshow(debug_img)
		ax[0].set_title("Input Image")
		ax[1].imshow(debug_mask, cmap='jet', vmin=0, vmax=20) # 21 classes
		ax[1].set_title("Ground Truth")
		ax[2].imshow(debug_pred, cmap='jet', vmin=0, vmax=20)
		ax[2].set_title(f"Prediction (Epoch {epoch+1})")
		
		plt.savefig(f'./debug_epoch_{epoch+1}.png')
		plt.close()

# ===== 🆕 訓練結束後畫圖 =====
print("\n" + "="*60)
print("Training Completed!")
print(f"Best Validation Dice: {best_dice:.4f} at Epoch {best_epoch}")
print("="*60)

# ===== 繪製訓練曲線 =====
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

epochs_range = range(1, num_epochs + 1)

# 1. Loss Curve
axes[0, 0].plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
axes[0, 0].plot(epochs_range, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Loss Curve', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 2. Dice Coefficient Curve
axes[0, 1].plot(epochs_range, history['train_dice'], 'b-o', label='Train Dice', linewidth=2, markersize=4)
axes[0, 1].plot(epochs_range, history['val_dice'], 'r-s', label='Val Dice', linewidth=2, markersize=4)
axes[0, 1].axhline(y=best_dice, color='g', linestyle='--', linewidth=2, label=f'Best: {best_dice:.4f}')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Dice Coefficient', fontsize=12)
axes[0, 1].set_title('Dice Coefficient Curve', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 🆕 3. IoU Curve
axes[1, 0].plot(epochs_range, history['train_iou'], 'b-o', label='Train IoU', linewidth=2, markersize=4)
axes[1, 0].plot(epochs_range, history['val_iou'], 'r-s', label='Val IoU', linewidth=2, markersize=4)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('IoU Score', fontsize=12)
axes[1, 0].set_title('IoU Curve', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# 4. Pixel Accuracy Curve
axes[1, 1].plot(epochs_range, history['pixel_acc'], 'g-^', label='Pixel Accuracy', linewidth=2, markersize=4)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Pixel Accuracy', fontsize=12)
axes[1, 1].set_title('Pixel Accuracy Curve', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./training_curves.png', dpi=300, bbox_inches='tight')
print("\n✓ Training curves saved to './training_curves.png'")
plt.close()

# ===== 🆕 印出最終統計 =====
print("\n" + "="*60)
print("Training Summary")
print("="*60)
print(f"Total Epochs: {num_epochs}")
print(f"Best Validation Dice: {best_dice:.4f} (Epoch {best_epoch})")
print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
print(f"Final Train Dice: {history['train_dice'][-1]:.4f}")
print(f"Final Val Dice: {history['val_dice'][-1]:.4f}")
print(f"Final Train IoU: {history['train_iou'][-1]:.4f}")
print(f"Final Val IoU: {history['val_iou'][-1]:.4f}")
print(f"Final Pixel Accuracy: {history['pixel_acc'][-1]:.4f}")

# 分析 overfitting
final_gap = history['train_dice'][-1] - history['val_dice'][-1]
if final_gap > 0.05:
	print(f"\n⚠️  Warning: Potential overfitting detected! (Gap: {final_gap:.4f})")
	print("    Consider: more data augmentation or early stopping")
elif final_gap < 0:
	print(f"\n⚠️  Warning: Val Dice > Train Dice (Gap: {final_gap:.4f})")
	print("    This might indicate: data leakage or unusual validation set")
else:
	print(f"\n✓ Good fit! Train-Val gap: {final_gap:.4f}")

print("="*60)

# ===== 🆕 儲存訓練歷史到 CSV =====
import pandas as pd

history_df = pd.DataFrame({
	'epoch': list(range(1, num_epochs + 1)),
	'train_loss': history['train_loss'],
	'val_loss': history['val_loss'],
	'train_dice': history['train_dice'],
	'val_dice': history['val_dice'],
	'train_iou': history['train_iou'],
	'val_iou': history['val_iou'],
	'pixel_acc': history['pixel_acc']
})

history_df.to_csv('./training_history.csv', index=False)
print("\n✓ Training history saved to './training_history.csv'")