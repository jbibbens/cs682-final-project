# train_gassl.py
import os
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from train_poverty import (
    set_seed,
    GatedAttention,
    TractRegressionModel,
    CityTractDataset,
    collate_bags,
)


def load_gassl_backbone(checkpoint_path, device):
    print(f"Loading GASSL checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    # MoCo-v2 stores encoder_q weights as 'module.encoder_q.*'.
    # Strip that prefix and drop the projection/fc head.
    encoder_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("module.encoder_q."):
            continue
        new_key = k[len("module.encoder_q."):]
        if new_key.startswith("fc."):
            continue
        encoder_dict[new_key] = v

    backbone = models.resnet50(weights=None)
    missing, unexpected = backbone.load_state_dict(encoder_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    # Replace the classifier with identity so forward() returns 2048-d features.
    feature_dim = backbone.fc.in_features  # 2048
    backbone.fc = nn.Identity()

    backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    print(f"GASSL ResNet-50 loaded. Feature dim: {feature_dim}")
    return backbone, feature_dim


def main():
    parser = argparse.ArgumentParser(description="Train poverty model with GASSL ResNet-50 backbone")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--gassl_checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--output", type=str, default="./best_model_gassl.pth")
    args = parser.parse_args()

    set_seed(42)
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

    backbone, feature_dim = load_gassl_backbone(args.gassl_checkpoint, device)

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
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=args.lr, weight_decay=args.wd)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_mae = float("inf")
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
