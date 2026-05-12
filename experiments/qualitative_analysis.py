"""
Qualitative analysis for the poverty prediction models.

Produces three analyses as discussed in the project proposal:
  1. Attention weight visualization  -- top/bottom weighted tiles per tract
  2. Prediction scatter plot         -- predicted vs. true poverty rates
  3. Grad-CAM heatmaps               -- spatial saliency for GASSL (ResNet-50)

Usage:
  python qualitative_analysis.py --backbone gassl   [--n_tracts 6]
  python qualitative_analysis.py --backbone satdino [--n_tracts 6]

Outputs are saved to qualitative/ directory.
"""

import os
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from train_poverty import (
    GatedAttention, TractRegressionModel, CityTractDataset,
    load_satdino_backbone, set_seed,
)
from train_gassl import load_gassl_backbone

OUT_DIR = "qualitative"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Modified GatedAttention that also returns per-tile weights
# ------------------------------------------------------------------
class GatedAttentionWithWeights(GatedAttention):
    def forward(self, h):
        v = torch.tanh(self.V(h))
        u = torch.sigmoid(self.U(h))
        combined = self.dropout(v * u)
        a = self.w(combined)
        a = torch.softmax(a, dim=0)
        return torch.sum(a * h, dim=0), a.squeeze(-1)


# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(args, device):
    if args.backbone == "satdino":
        backbone, feature_dim = load_satdino_backbone(args.backbone_checkpoint, device)
    else:
        backbone, feature_dim = load_gassl_backbone(args.backbone_checkpoint, device)

    model = TractRegressionModel(backbone, feature_dim,
                                 attn_dropout=0.2, final_dropout=0.2)
    # Swap in the weight-returning attention module
    model.attention = GatedAttentionWithWeights(
        feature_dim, hidden_dim=128, dropout=0.2
    ).to(device)
    model.load_state_dict(
        torch.load(args.model_checkpoint, map_location=device, weights_only=False),
        strict=False,
    )
    model.to(device)
    model.eval()
    return model, feature_dim


def get_attention_weights(model, images_tensor, device):
    """Run one tract through backbone + attention, return (pred, weights)."""
    with torch.no_grad():
        feats = model.backbone(images_tensor.to(device))
        emb, weights = model.attention(feats)
        emb = model.norm(emb.unsqueeze(0))
        pred = torch.sigmoid(model.regressor(emb)).item()
    return pred, weights.cpu().numpy()


# ------------------------------------------------------------------
# 1. Attention weight visualisation
# ------------------------------------------------------------------
def visualise_attention(args, model, dataset, device, n_tracts=6):
    """For n_tracts, show top-3 and bottom-3 weighted tiles side by side."""
    indices = random.sample(range(len(dataset)), min(n_tracts, len(dataset)))

    for idx in indices:
        images_pil, label = dataset[idx]
        city, cbg = dataset.tract_ids[idx]

        images_tensor = torch.stack([val_transform(img) for img in images_pil])
        pred, weights = get_attention_weights(model, images_tensor, device)

        order = np.argsort(weights)[::-1]
        top_k    = order[:3]
        bottom_k = order[-3:][::-1]

        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        fig.suptitle(
            f"{city} — CBG {cbg}\n"
            f"True poverty: {label:.3f}   Predicted: {pred:.3f}",
            fontsize=11, fontweight="bold"
        )

        for col, tile_idx in enumerate(top_k):
            ax = axes[0, col]
            ax.imshow(images_pil[tile_idx])
            ax.set_title(f"Top {col+1}  (w={weights[tile_idx]:.3f})", fontsize=9)
            ax.axis("off")

        for col, tile_idx in enumerate(bottom_k):
            ax = axes[1, col]
            ax.imshow(images_pil[tile_idx])
            ax.set_title(f"Bottom {col+1}  (w={weights[tile_idx]:.3f})", fontsize=9)
            ax.axis("off")

        axes[0, 0].set_ylabel("High attention", fontsize=9, labelpad=4)
        axes[1, 0].set_ylabel("Low attention",  fontsize=9, labelpad=4)

        fname = os.path.join(OUT_DIR, f"attention_{args.backbone}_{city}_{cbg}.png")
        fig.tight_layout()
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")


# ------------------------------------------------------------------
# 2. Prediction scatter plot
# ------------------------------------------------------------------
def prediction_scatter(args):
    import pandas as pd

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=130)
    fig.suptitle("Predicted vs. True Poverty Rate — Los Angeles Test Set",
                 fontsize=12, fontweight="bold")

    configs = [
        ("satdino", "checkpoints/test_predictions_satdino.csv", "#2563EB"),
        ("gassl",   "checkpoints/test_predictions_gassl.csv",   "#DC2626"),
    ]

    for ax, (name, csv_path, color) in zip(axes, configs):
        if not os.path.exists(csv_path):
            ax.set_title(f"{name.upper()} (no predictions file)")
            continue

        df = pd.read_csv(csv_path)
        true, pred = df["true"].values, df["pred"].values

        mae  = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2   = 1 - ss_res / ss_tot

        ax.scatter(true, pred, color=color, alpha=0.55, s=18, edgecolors="none")
        lims = [min(true.min(), pred.min()) - 0.02,
                max(true.max(), pred.max()) + 0.02]
        ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("True Poverty Rate", fontsize=10)
        ax.set_ylabel("Predicted Poverty Rate", fontsize=10)
        ax.set_title(f"{name.upper()}", fontsize=11, fontweight="bold")
        ax.text(0.05, 0.92,
                f"MAE={mae:.4f}  RMSE={rmse:.4f}  $R^2$={r2:.3f}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "prediction_scatter.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ------------------------------------------------------------------
# 3. Grad-CAM (GASSL / ResNet-50 only)
# ------------------------------------------------------------------
def gradcam(args, model, dataset, device, n_tiles=6):
    """Compute Grad-CAM on the last ResNet conv layer for sample tiles."""
    if args.backbone != "gassl":
        print("Grad-CAM is only supported for the GASSL (ResNet-50) backbone.")
        return

    # Hook storage
    activations, gradients = {}, {}

    def fwd_hook(module, inp, out):
        activations["value"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    # Last ResNet conv block
    target_layer = model.backbone.layer4[-1].conv3
    fwd_h = target_layer.register_forward_hook(fwd_hook)
    bwd_h = target_layer.register_full_backward_hook(bwd_hook)

    indices = random.sample(range(len(dataset)), min(n_tiles, len(dataset)))
    all_tiles, all_cams, all_labels, all_preds = [], [], [], []

    for idx in indices:
        images_pil, label = dataset[idx]
        # Use highest-attention tile only
        images_tensor = torch.stack([val_transform(img) for img in images_pil])
        _, weights = get_attention_weights(model, images_tensor, device)
        best_tile_idx = int(np.argmax(weights))
        tile_tensor = val_transform(images_pil[best_tile_idx]).unsqueeze(0).to(device)

        model.zero_grad()
        feat = model.backbone(tile_tensor)  # (1, 2048)
        # Scalar output for Grad-CAM
        out = torch.sigmoid(model.regressor(model.norm(feat)))
        out.backward()

        act  = activations["value"].squeeze(0)   # (C, H, W)
        grad = gradients["value"].squeeze(0)      # (C, H, W)
        weights_cam = grad.mean(dim=(1, 2))       # (C,)
        cam = torch.relu((weights_cam[:, None, None] * act).sum(dim=0))
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        all_tiles.append(images_pil[best_tile_idx])
        all_cams.append(cam)
        all_labels.append(label)
        all_preds.append(out.item())

    fwd_h.remove()
    bwd_h.remove()

    n = len(all_tiles)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    fig.suptitle("Grad-CAM on Highest-Attention Tile per Tract (GASSL)",
                 fontsize=11, fontweight="bold")

    for i in range(n):
        tile_np = np.array(all_tiles[i].resize((224, 224)))
        cam_up  = np.array(
            Image.fromarray((all_cams[i] * 255).astype(np.uint8)).resize(
                (224, 224), Image.BILINEAR
            )
        ) / 255.0

        axes[0, i].imshow(tile_np)
        axes[0, i].set_title(f"True={all_labels[i]:.2f}\nPred={all_preds[i]:.2f}",
                              fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(tile_np)
        axes[1, i].imshow(cam_up, cmap="jet", alpha=0.45)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM",  fontsize=9)

    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "gradcam_gassl.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",            type=str, required=True,
                        choices=["satdino", "gassl"])
    parser.add_argument("--backbone_checkpoint", type=str,
                        default=None)
    parser.add_argument("--model_checkpoint",    type=str,
                        default=None)
    parser.add_argument("--test_dir",            type=str,
                        default="satellite_imagery_collection/data_dir/test")
    parser.add_argument("--labels_dir",          type=str,
                        default="labels")
    parser.add_argument("--n_tracts",            type=int, default=6)
    parser.add_argument("--seed",                type=int, default=42)
    args = parser.parse_args()

    # Defaults
    if args.backbone_checkpoint is None:
        args.backbone_checkpoint = (
            "checkpoints/satdino-vit_small-16.pth" if args.backbone == "satdino"
            else "checkpoints/gassl_mocov2_tp_resnet50.pth.tar"
        )
    if args.model_checkpoint is None:
        args.model_checkpoint = (
            "checkpoints/best_model.pth" if args.backbone == "satdino"
            else "checkpoints/best_model_gassl.pth"
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Backbone: {args.backbone}")

    model, _ = load_model(args, device)

    dataset = CityTractDataset(args.test_dir, args.labels_dir,
                               transform=None)  # raw PIL for visualisation

    print("\n--- 1. Attention weight visualisation ---")
    visualise_attention(args, model, dataset, device, n_tracts=args.n_tracts)

    print("\n--- 2. Prediction scatter plot ---")
    prediction_scatter(args)

    print("\n--- 3. Grad-CAM (GASSL only) ---")
    gradcam(args, model, dataset, device, n_tiles=args.n_tracts)

    print(f"\nAll outputs saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
