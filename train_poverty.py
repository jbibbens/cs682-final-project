# train_poverty.py
import os
import math
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------------------
# Gated Attention Module
# ---------------------------
class GatedAttention(nn.Module):
    """
    Gated attention mechanism from Ilse et al. (ICML 2018)
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        v = torch.tanh(self.V(h))
        u = torch.sigmoid(self.U(h))
        combined = self.dropout(v * u)
        a = self.w(combined)
        a = torch.softmax(a, dim=0)
        return torch.sum(a * h, dim=0)

# ---------------------------
# Full Model
# ---------------------------
class TractRegressionModel(nn.Module):
    def __init__(self, backbone, feature_dim, attn_dropout=0.7, final_dropout=0.7):
        super().__init__()
        self.backbone = backbone
        self.attention = GatedAttention(feature_dim, hidden_dim=128, dropout=attn_dropout)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(final_dropout)
        self.regressor = nn.Linear(feature_dim, 1)
        nn.init.kaiming_normal_(self.regressor.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.regressor.bias)

    def forward(self, all_images, lengths):
        with torch.no_grad():
            features = self.backbone(all_images)
        feature_list = torch.split(features, lengths, dim=0)
        tract_embs = []
        for feats in feature_list:
            emb = self.attention(feats)
            tract_embs.append(emb)
        tract_embs = torch.stack(tract_embs, dim=0)
        tract_embs = self.norm(tract_embs)
        tract_embs = self.dropout(tract_embs)
        preds = torch.sigmoid(self.regressor(tract_embs).squeeze(-1))
        return preds

# ---------------------------
# Dataset
# ---------------------------
class CityTractDataset(Dataset):
    """
    Directory layout:
        data_dir/
          {city}/
            {cbg_code}/
              img1.png ...

    labels_dir/
        {city}.csv   -- columns: filename, CBG Code, Poverty Rate
    """
    def __init__(self, data_dir, labels_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # (city, cbg_code) -> poverty_rate
        self.labels = {}
        # list of (city, cbg_code) pairs that exist on disk
        self.tract_ids = []

        for city in sorted(os.listdir(data_dir)):
            city_path = os.path.join(data_dir, city)
            if not os.path.isdir(city_path):
                continue

            label_csv = os.path.join(labels_dir, f"{city}.csv")
            if not os.path.isfile(label_csv):
                print(f"Warning: no label CSV for {city}, skipping.")
                continue

            df = pd.read_csv(label_csv)
            # build CBG Code -> poverty rate mapping (one rate per CBG)
            cbg_to_rate = (
                df.groupby("CBG Code")["Poverty Rate"].first().to_dict()
            )

            for cbg_dir in sorted(os.listdir(city_path)):
                tract_path = os.path.join(city_path, cbg_dir)
                if not os.path.isdir(tract_path):
                    continue
                cbg_code = int(cbg_dir)
                if cbg_code not in cbg_to_rate:
                    print(f"Warning: {city}/{cbg_dir} has no label, skipping.")
                    continue
                self.labels[(city, cbg_dir)] = cbg_to_rate[cbg_code]
                self.tract_ids.append((city, cbg_dir))

        print(f"Loaded {len(self.tract_ids)} tracts from {data_dir}")

    def __len__(self):
        return len(self.tract_ids)

    def __getitem__(self, idx):
        city, cbg_dir = self.tract_ids[idx]
        label = self.labels[(city, cbg_dir)]
        tract_dir = os.path.join(self.data_dir, city, cbg_dir)
        images = []
        for fname in sorted(os.listdir(tract_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(tract_dir, fname)).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
        if not images:
            raise RuntimeError(f"No images in {tract_dir}")
        return images, torch.tensor(label, dtype=torch.float32)

def collate_bags(batch):
    all_images, lengths, labels = [], [], []
    for images, label in batch:
        lengths.append(len(images))
        all_images.extend(images)
        labels.append(label)
    return torch.stack(all_images), lengths, torch.stack(labels)

# ---------------------------
# Backbone loading for SatDINO
# ---------------------------
def load_satdino_backbone(checkpoint_path, device):
    print(f"Loading SatDINO checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'student' in checkpoint:
        state_dict = checkpoint['student']
        print("Using 'student' weights from checkpoint.")
    elif 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
        print("Using 'teacher' weights from checkpoint.")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    backbone_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith('head') and not k.startswith('projection')
    }

    # SatDINO adds a gsd_register token, making pos_embed [1, 198, 384].
    # Standard ViT-Small expects [1, 197, 384], so drop the extra register slot (index 1).
    if 'pos_embed' in backbone_dict:
        pe = backbone_dict['pos_embed']  # [1, 198, 384]
        if pe.shape[1] == 198:
            backbone_dict['pos_embed'] = torch.cat([pe[:, :1], pe[:, 2:]], dim=1)  # [1, 197, 384]

    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)
    missing, unexpected = model.load_state_dict(backbone_dict, strict=False)
    if missing or unexpected:
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    feature_dim = model.embed_dim  # 384 for ViT-Small
    print(f"SatDINO loaded. Feature dim: {feature_dim}")
    return model, feature_dim

# ---------------------------
# Main training
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train poverty prediction model with SatDINO backbone")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Root directory of pre-split training data: {city}/{cbg_code}/images")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Root directory of pre-split validation data: {city}/{cbg_code}/images")
    parser.add_argument("--labels_dir", type=str, required=True,
                        help="Directory containing per-city CSVs: {city}.csv")
    parser.add_argument("--satdino_checkpoint", type=str, required=True,
                        help="Path to SatDINO checkpoint (satdino-vit_small-16.pth)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--output", type=str, default="./best_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    backbone, feature_dim = load_satdino_backbone(args.satdino_checkpoint, device)

    train_dataset = CityTractDataset(args.train_dir, args.labels_dir, transform=train_transform)
    val_dataset   = CityTractDataset(args.val_dir,   args.labels_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_bags, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_bags, num_workers=4, pin_memory=True)

    model = TractRegressionModel(backbone, feature_dim,
                                 attn_dropout=args.dropout, final_dropout=args.dropout)
    model.to(device)

    criterion = nn.MSELoss()
    # backbone is frozen, so only attention + regressor params are trained
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=args.lr, weight_decay=args.wd)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_mae = float('inf')
    epochs_no_improve = 0
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for all_images, lengths, labels in train_loader:
            all_images = all_images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(all_images, lengths)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for all_images, lengths, labels in val_loader:
                all_images = all_images.to(device)
                labels = labels.to(device)
                preds = model(all_images, lengths)
                val_mae += torch.abs(preds - labels).sum().item()
        val_mae /= len(val_loader.dataset)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.4f} | LR: {current_lr:.2e}", flush=True)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.output)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Training complete. Best validation MAE: {best_val_mae:.4f}")
    print(f"Best model saved to {args.output}")

if __name__ == "__main__":
    main()
