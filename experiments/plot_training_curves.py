"""
Plots Train Loss and Val MAE curves for SatDINO and GASSL runs side by side.
Usage: python plot_training_curves.py
Output: logs/training_curves.png
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOGS = {
    "SatDINO (ViT-S/16)": "logs/train_56555783.out",
    "GASSL (ResNet-50)":   "logs/train_gassl_56557562.out",
}
OUT = "logs/training_curves.png"

PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([\d.]+)\s+\|\s+Val MAE:\s+([\d.]+)"
)


def parse_log(path):
    epochs, train_loss, val_mae = [], [], []
    with open(path) as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_mae.append(float(m.group(3)))
    return epochs, train_loss, val_mae


COLORS = {
    "SatDINO (ViT-S/16)": "#2563EB",
    "GASSL (ResNet-50)":   "#DC2626",
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
fig.suptitle("Training Curves: SatDINO vs GASSL", fontsize=13, fontweight="bold", y=1.01)

ax_loss, ax_mae = axes

for name, path in LOGS.items():
    epochs, train_loss, val_mae = parse_log(path)
    color = COLORS[name]
    ax_loss.plot(epochs, train_loss, color=color, linewidth=2, marker="o",
                 markersize=3, label=name)
    ax_mae.plot(epochs, val_mae, color=color, linewidth=2, marker="o",
                markersize=3, label=name)

for ax, title, ylabel in [
    (ax_loss, "Train Loss (MSE)", "MSE"),
    (ax_mae,  "Validation MAE",   "MAE"),
]:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved: {OUT}")
