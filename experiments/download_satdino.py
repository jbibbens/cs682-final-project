# download_satdino.py
import os
import argparse
from huggingface_hub import hf_hub_download

def download_satdino(output_dir="./pretrained_models"):
    """
    Download SatDINO ViT-Small/16 weights from Hugging Face Hub.
    The checkpoint file (satdino-vit_small-16.pth) is saved in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Official repository and filename
    repo_id = "strakajk/satdino-vit_small-16"
    filename = "satdino-vit_small-16.pth"   # adjust if the filename differs

    print(f"Downloading SatDINO ViT-Small/16 from {repo_id} ...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=output_dir,
        force_download=True,      # set to False to use cached copy
    )

    # hf_hub_download places files inside a subdirectory structure by default.
    import shutil
    dest = os.path.join(output_dir, filename)
    if local_path != dest:
        shutil.copy(local_path, dest)
        print(f"Copied checkpoint to {dest}")
    print(f"SatDINO checkpoint ready at {dest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SatDINO ViT-Small/16")
    parser.add_argument("--output_dir", default="./pretrained_models",
                        help="Where to save the model checkpoint")
    args = parser.parse_args()
    download_satdino(args.output_dir)