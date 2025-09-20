import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from unet import UNet
from deeplabv3_plus import DeepLabV3Plus
import torch.nn.functional as F
from hrnet import HRNetSegmentation

# Dataset personalizado
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size):
        self.image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.png')])
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_files[idx])
        img = cv2.resize(img, self.image_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

# Função métrica IoU
def iou_score(preds, labels, num_classes, smooth=1e-6):
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)
        intersection = (pred_cls & label_cls).float().sum()
        union = (pred_cls | label_cls).float().sum()
        ious.append((intersection + smooth) / (union + smooth))
    return torch.mean(torch.stack(ious))

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# Boundary Loss (simplified using edge detection)
def compute_boundary_mask(mask):
    laplace_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
    edges = F.conv2d(mask.unsqueeze(1).float(), laplace_kernel, padding=1).abs()
    return (edges > 0).float()

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplace_kernel', kernel)

    def forward(self, preds, targets):
        preds = torch.argmax(preds, dim=1, keepdim=True).float()
        targets = targets.unsqueeze(1).float()

        # Garante que kernel está no mesmo device que o input
        kernel = self.laplace_kernel.to(preds.device)

        pred_boundary = F.conv2d(preds, kernel, padding=1).abs() > 0
        target_boundary = F.conv2d(targets, kernel, padding=1).abs() > 0
        loss = F.binary_cross_entropy(pred_boundary.float(), target_boundary.float())
        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss




if __name__ == "__main__":

    torch.cuda.empty_cache()

    # Configurações
    input_shape = (512, 512)
    n_labels = 7
    batch_size = 4
    epochs = 300
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Carregamento dos dados
    train_dataset = SegmentationDataset('../../Data/Train/images', '../../Data/Train/Masks', input_shape)
    val_dataset = SegmentationDataset('../../Data/Val/images', '../../Data/Val/Masks', input_shape)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False, num_workers=0)

    # Cálculo de pesos de classe
    class_counts = torch.zeros(n_labels)
    for _, mask in train_dataset:
        unique, counts = torch.unique(mask, return_counts=True)
        class_counts[unique] += counts
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()

    # Modelo, losses, otimizador e scheduler
    #model = HRNetSegmentation(num_classes=n_labels).to(device)
    model = DeepLabV3Plus(num_classes=n_labels).to(device)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))
    dice_loss = DiceLoss()
    boundary_loss = BoundaryLoss()
    focal_loss = FocalLoss(weight=class_weights.to(device))



    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)

    # Treino
    best_val_loss = float('inf')
    best_val_iou = 0.0
    early_stop_counter = 0
    patience = 40

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = 0.7 * ce_loss(outputs, masks) + 0.3 * dice_loss(outputs, masks) + 0.4 * boundary_loss(outputs, masks) + 0.5 * focal_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = 0.7 * ce_loss(outputs, masks) + 0.3 * dice_loss(outputs, masks) + 0.4 * boundary_loss(outputs, masks) + 0.5 * focal_loss(outputs, masks)
                val_loss += loss.detach().item()
                val_iou += iou_score(outputs, masks, n_labels).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), "best_iou_model.pth")
            print("Best IoU model saved.")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    torch.save(model.state_dict(), "final_model.pth")
    print("Final model saved.")
