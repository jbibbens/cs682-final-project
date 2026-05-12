"""
Li-paper-style qualitative visualizations:
  Figure A: Gallery of high-attention tiles (across many tracts)
  Figure B: Gallery of low-attention tiles (across many tracts)
  Figure C: Grad-CAM overlays on highest-attention tiles (GASSL only)

Usage:
  python plot_qualitative.py --backbone gassl   [--n_tracts 30]
  python plot_qualitative.py --backbone satdino [--n_tracts 30]
"""

import os, argparse, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms as T

from train_poverty import (
    GatedAttention, TractRegressionModel, CityTractDataset,
    load_satdino_backbone, set_seed,
)
from train_gassl import load_gassl_backbone

OUT_DIR = "qualitative"
os.makedirs(OUT_DIR, exist_ok=True)

val_tf = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ------------------------------------------------------------------ #
# Attention module that returns weights                               #
# ------------------------------------------------------------------ #
class GatedAttentionWithWeights(GatedAttention):
    def forward(self, h):
        v = torch.tanh(self.V(h))
        u = torch.sigmoid(self.U(h))
        a = self.w(self.dropout(v * u))
        a = torch.softmax(a, dim=0)
        return torch.sum(a * h, dim=0), a.squeeze(-1)


# ------------------------------------------------------------------ #
# Model loading                                                       #
# ------------------------------------------------------------------ #
def load_model(args, device):
    if args.backbone == "satdino":
        backbone, fdim = load_satdino_backbone(args.backbone_checkpoint, device)
    else:
        backbone, fdim = load_gassl_backbone(args.backbone_checkpoint, device)

    model = TractRegressionModel(backbone, fdim, attn_dropout=0.2, final_dropout=0.2)
    model.attention = GatedAttentionWithWeights(fdim, hidden_dim=128, dropout=0.2).to(device)
    model.load_state_dict(
        torch.load(args.model_checkpoint, map_location=device, weights_only=False),
        strict=False,
    )
    model.to(device).eval()
    return model, fdim


# ------------------------------------------------------------------ #
# Per-tract inference: returns (pred, weights, pil_images)           #
# ------------------------------------------------------------------ #
def infer_tract(model, images_pil, device):
    imgs_t = torch.stack([val_tf(img) for img in images_pil]).to(device)
    with torch.no_grad():
        feats = model.backbone(imgs_t)
        emb, weights = model.attention(feats)
        emb = model.norm(emb.unsqueeze(0))
        pred = torch.sigmoid(model.regressor(emb)).item()
    return pred, weights.cpu().numpy()


# ------------------------------------------------------------------ #
# Figure A & B: attention weight galleries                           #
# ------------------------------------------------------------------ #
def attention_gallery(args, model, dataset, device, n_tracts=30, n_show=3):
    """Collect top-1 and bottom-1 attention tile per tract, then display
    a 3×3 gallery for high-attention and low-attention tiles (Fig 6 & 7)."""
    indices = random.sample(range(len(dataset)), min(n_tracts, len(dataset)))

    high_tiles, low_tiles, high_meta, low_meta = [], [], [], []

    for idx in indices:
        images_pil, label = dataset[idx]
        if len(images_pil) < 2:
            continue
        city, cbg = dataset.tract_ids[idx]
        pred, weights = infer_tract(model, images_pil, device)
        order = np.argsort(weights)[::-1]

        high_tiles.append(images_pil[order[0]])
        high_meta.append(f"w={weights[order[0]]:.3f}\ntrue={label:.2f} pred={pred:.2f}")

        low_tiles.append(images_pil[order[-1]])
        low_meta.append(f"w={weights[order[-1]]:.3f}\ntrue={label:.2f} pred={pred:.2f}")

    for tiles, meta, tag in [
        (high_tiles, high_meta, "high"),
        (low_tiles,  low_meta,  "low"),
    ]:
        tiles = tiles[:n_show]
        meta  = meta[:n_show]
        cols  = 3
        rows  = (len(tiles) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 5.0))
        axes = np.array(axes).flatten()

        title = ("Examples of Tiles Given Higher Attention Weights"
                 if tag == "high" else
                 "Examples of Tiles Given Lower Attention Weights")
        fig.suptitle(f"{title}\n({args.backbone.upper()})",
                     fontsize=14, fontweight="bold")

        for i, (tile, m) in enumerate(zip(tiles, meta)):
            axes[i].imshow(tile)
            axes[i].set_title(m, fontsize=12)
            axes[i].axis("off")
        for j in range(len(tiles), len(axes)):
            axes[j].axis("off")

        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f"attention_gallery_{args.backbone}_{tag}.png")
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")


# ------------------------------------------------------------------ #
# Figure C: saliency gallery — Grad-CAM (GASSL) or gradient         #
#           saliency (SatDINO ViT, which has no conv target layer)   #
# ------------------------------------------------------------------ #
def gradcam_gallery(args, model, dataset, device, n_tracts=30, n_show=3):
    activations, gradients = {}, {}

    if args.backbone == "gassl":
        def fwd_hook(_, __, out):
            activations["v"] = out

        def bwd_hook(_, __, grad_out):
            gradients["v"] = grad_out[0]

        target = model.backbone.layer4[-1].conv3
        fh = target.register_forward_hook(fwd_hook)
        bh = target.register_full_backward_hook(bwd_hook)

    # Temporarily unfreeze backbone so gradients flow
    for p in model.backbone.parameters():
        p.requires_grad_(True)

    indices = random.sample(range(len(dataset)), min(n_tracts, len(dataset)))
    results = []

    for idx in indices:
        images_pil, label = dataset[idx]
        if len(images_pil) < 1:
            continue
        pred, weights = infer_tract(model, images_pil, device)
        best = int(np.argmax(weights))
        tile_t = val_tf(images_pil[best]).unsqueeze(0).to(device)

        model.zero_grad()

        if args.backbone == "gassl":
            with torch.enable_grad():
                feat = model.backbone(tile_t)
                out  = torch.sigmoid(model.regressor(model.norm(feat)))
                out.backward()

            act = activations["v"].squeeze(0)
            grd = gradients["v"].squeeze(0)
            cam = torch.relu((grd.mean(dim=(1, 2), keepdim=True) * act).sum(dim=0))
            cam = cam.cpu().detach().numpy()
        else:
            # Gradient saliency: |d output / d input| max-pooled over channels
            tile_grad = tile_t.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                feat = model.backbone(tile_grad)
                out  = torch.sigmoid(model.regressor(model.norm(feat)))
                out.backward()
            cam = tile_grad.grad.abs().max(dim=1)[0].squeeze(0).cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        results.append((images_pil[best], cam, label, pred))
        if len(results) >= n_show:
            break

    if args.backbone == "gassl":
        fh.remove(); bh.remove()
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    overlay_label = "Grad-CAM Overlay" if args.backbone == "gassl" else "Gradient Saliency"
    backbone_name = args.backbone.upper()
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(n * 4.5, 9.0))
    fig.suptitle(
        f"{overlay_label} on Highest-Attention Tile per Tract ({backbone_name})",
        fontsize=13, fontweight="bold",
    )

    for i, (tile, cam, label, pred) in enumerate(results):
        tile_np = np.array(tile.resize((224, 224)))
        cam_up  = np.array(
            Image.fromarray((cam * 255).astype(np.uint8))
            .resize((224, 224), Image.BILINEAR)
        ) / 255.0

        axes[0, i].imshow(tile_np)
        axes[0, i].set_title(f"true={label:.2f}\npred={pred:.2f}", fontsize=11)
        axes[0, i].axis("off")

        axes[1, i].imshow(tile_np)
        axes[1, i].imshow(cam_up, cmap="jet", alpha=0.45)
        axes[1, i].set_title(overlay_label, fontsize=11)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original tile", fontsize=10)
    axes[1, 0].set_ylabel(overlay_label,   fontsize=10)

    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f"gradcam_gallery_{args.backbone}.png")
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",            required=True, choices=["satdino","gassl"])
    parser.add_argument("--backbone_checkpoint", default=None)
    parser.add_argument("--model_checkpoint",    default=None)
    parser.add_argument("--test_dir",  default="satellite_imagery_collection/data_dir/test")
    parser.add_argument("--labels_dir",default="labels")
    parser.add_argument("--n_tracts",  type=int, default=30)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

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
    print(f"Device: {device}  Backbone: {args.backbone}")

    model, _ = load_model(args, device)
    dataset  = CityTractDataset(args.test_dir, args.labels_dir, transform=None)

    print("\n--- Attention galleries ---")
    attention_gallery(args, model, dataset, device, n_tracts=args.n_tracts)

    print("\n--- Grad-CAM gallery ---")
    gradcam_gallery(args, model, dataset, device, n_tracts=args.n_tracts)

    print(f"\nAll outputs in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
