# eval_poverty.py
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from train_poverty import (
    GatedAttention,
    TractRegressionModel,
    CityTractDataset,
    collate_bags,
    load_satdino_backbone,
    set_seed,
)
from train_gassl import load_gassl_backbone


def main():
    parser = argparse.ArgumentParser(description="Evaluate poverty prediction model on test set")
    parser.add_argument("--test_dir",         type=str,   required=True)
    parser.add_argument("--labels_dir",       type=str,   required=True)
    parser.add_argument("--model_checkpoint", type=str,   required=True)
    parser.add_argument("--backbone",         type=str,   default="satdino",
                        choices=["satdino", "gassl"],
                        help="Which backbone the model checkpoint was trained with")
    parser.add_argument("--satdino_checkpoint", type=str, default=None,
                        help="Required when --backbone satdino")
    parser.add_argument("--gassl_checkpoint",   type=str, default=None,
                        help="Required when --backbone gassl")
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--dropout",          type=float, default=0.2)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if args.backbone == "satdino":
        if not args.satdino_checkpoint:
            raise ValueError("--satdino_checkpoint is required when --backbone satdino")
        backbone, feature_dim = load_satdino_backbone(args.satdino_checkpoint, device)
    else:
        if not args.gassl_checkpoint:
            raise ValueError("--gassl_checkpoint is required when --backbone gassl")
        backbone, feature_dim = load_gassl_backbone(args.gassl_checkpoint, device)

    model = TractRegressionModel(backbone, feature_dim,
                                 attn_dropout=args.dropout, final_dropout=args.dropout)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device,
                                     weights_only=False))
    model.to(device)
    model.eval()
    print(f"Loaded {args.backbone.upper()} model from {args.model_checkpoint}")

    test_dataset = CityTractDataset(args.test_dir, args.labels_dir, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_bags, num_workers=4, pin_memory=True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for all_images, lengths, labels in test_loader:
            all_images = all_images.to(device)
            preds = model(all_images, lengths)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mae  = np.mean(np.abs(all_preds - all_labels))
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    ss_res = np.sum((all_labels - all_preds) ** 2)
    ss_tot = np.sum((all_labels - all_labels.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n=== Test Results ({args.backbone.upper()}) ===")
    print(f"  Samples : {len(all_labels)}")
    print(f"  MAE     : {mae:.4f}")
    print(f"  RMSE    : {rmse:.4f}")
    print(f"  R²      : {r2:.4f}")

    out_csv = os.path.join(os.path.dirname(args.model_checkpoint),
                           f"test_predictions_{args.backbone}.csv")
    pd.DataFrame({"true": all_labels, "pred": all_preds}).to_csv(out_csv, index=False)
    print(f"\nPredictions saved to {out_csv}")


if __name__ == "__main__":
    main()
