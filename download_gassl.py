# download_gassl.py
# Downloads the GASSL MoCo-v2+TP ResNet-50 checkpoint pretrained on fMoW.
# Source: https://github.com/sustainlab-group/geography-aware-ssl
import os
import urllib.request

URL = "https://zenodo.org/record/7379715/files/moco_tp.pth.tar?download=1"
OUT = os.path.join(os.path.dirname(__file__), "checkpoints", "gassl_mocov2_tp_resnet50.pth.tar")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

if os.path.exists(OUT):
    print(f"Already exists: {OUT}")
else:
    print(f"Downloading GASSL MoCo-v2+TP checkpoint to {OUT} ...")
    urllib.request.urlretrieve(URL, OUT)
    print("Done.")

# Inspect checkpoint keys so the training script can load it correctly.
import torch
ckpt = torch.load(OUT, map_location="cpu", weights_only=False)
print("Top-level keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    keys = list(ckpt["state_dict"].keys())
    print("state_dict keys (first 5):", keys[:5])
