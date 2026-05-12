# train_poverty.py
import os
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
        self.dropout = nn.Dropout(final_dropout)
        self.regressor = nn.Linear(feature_dim, 1)

    def forward(self, all_images, lengths):
        with torch.no_grad():
            features = self.backbone(all_images)
        feature_list = torch.split(features, lengths, dim=0)
        tract_embs = []
        for feats in feature_list:
            emb = self.attention(feats)
            tract_embs.append(emb)
        tract_embs = torch.stack(tract_embs, dim=0)
        tract_embs = self.dropout(tract_embs)
        preds = self.regressor(tract_embs).squeeze(-1)
        return preds

# ---------------------------
# Dataset
# ---------------------------
class CityTractDataset(Dataset):
    """
    Expects:
        data_dir/
          City1/
            tract_123/
              img1.png
            tract_456/ ...
          City2/ ...
    labels_csv: columns city, tract_id, poverty_rate
    """
    def __init__(self, data_dir, labels_csv, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.labels_df = pd.read_csv(labels_csv)
        self.labels_df['tract_id'] = self.labels_df['tract_id'].astype(str)
        self.labels = {}
        self.tract_ids = []

        for _, row in self.labels_df.iterrows():
            city = row['city']
            tract_id = row['tract_id']
            tract_path = os.path.join(data_dir, city, tract_id)
            if os.path.isdir(tract_path):
                self.labels[(city, tract_id)] = row['poverty_rate']
                self.tract_ids.append((city, tract_id))
            else:
                print(f"Warning: tract path {tract_path} not found, skipping.")

    def __len__(self):
        return len(self.tract_ids)

    def __getitem__(self, idx):
        city, tract_id = self.tract_ids[idx]
        label = self.labels[(city, tract_id)]
        tract_dir = os.path.join(self.data_dir, city, tract_id)
        images = []
        for fname in sorted(os.listdir(tract_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(tract_dir, fname)
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
        if not images:
            raise RuntimeError(f"No images found in {tract_dir}")
        return images, torch.tensor(label, dtype=torch.float32)

def collate_bags(batch):
    all_images = []
    lengths = []
    labels = []
    for images, label in batch:
        lengths.append(len(images))
        all_images.extend(images)
        labels.append(label)
    all_images = torch.stack(all_images, dim=0)
    labels = torch.stack(labels, dim=0)
    return all_images, lengths, labels

# ---------------------------
# Backbone loading for SatDINO
# ---------------------------
def load_satdino_backbone(checkpoint_path, device):
    """
    Load the SatDINO ViT-Small/16 backbone from the downloaded checkpoint.
    The checkpoint typically contains DINO teacher/student and projection heads.
    We extract the student backbone weights and return a ViT feature extractor.
    """
    print(f"Loading SatDINO checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # The checkpoint could be a dict with 'student', 'teacher', 'state_dict', etc.
    if 'student' in checkpoint:
        state_dict = checkpoint['student']
        print("Found 'student' key in checkpoint, extracting student weights.")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove projection head keys (typically 'head' or if DINO, 'head.last_layer')
    backbone_dict = {}
    for k, v in state_dict.items():
        if not k.startswith('head') and not k.startswith('projection'):
            backbone_dict[k] = v

    # Create the ViT-S/16 model
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)
    missing, unexpected = model.load_state_dict(backbone_dict, strict=False)

    if missing or unexpected:
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
        print("Some keys mismatched; ensure the checkpoint is for ViT-S/16.")

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False  # frozen feature extractor

    feature_dim = model.embed_dim  # 384 for ViT-Small
    print(f"SatDINO backbone loaded. Feature dimension: {feature_dim}")
    return model, feature_dim

# ---------------------------
# Main training
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train poverty prediction model with SatDINO backbone")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory with city folders")
    parser.add_argument("--labels", type=str, required=True,
                        help="CSV file with columns: city, tract_id, poverty_rate")
    parser.add_argument("--satdino_checkpoint", type=str, required=True,
                        help="Path to downloaded SatDINO checkpoint (satdino-vit_small-16.pth)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--output", type=str, default="./best_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- Transforms (Verify SatDINO normalization from its documentation) -----
    # Most satellite-specific models trained on RGB have slightly different stats.
    # If SatDINO provides its own values, replace these.
    # For now we keep ImageNet defaults (common fallback).
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

    # ----- Load SatDINO backbone -----
    backbone, feature_dim = load_satdino_backbone(args.satdino_checkpoint, device)

    # ----- Datasets -----
    full_dataset = CityTractDataset(args.data_dir, args.labels, transform=train_transform)
    n_total = len(full_dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    train_dataset, val_dataset, _ = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_total - n_train - n_val]
    )
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_bags, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_bags, num_workers=4, pin_memory=True)

    # ----- Model, loss, optimizer -----
    model = TractRegressionModel(backbone, feature_dim,
                                 attn_dropout=args.dropout, final_dropout=args.dropout)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # ----- Training -----
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

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.4f}")

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