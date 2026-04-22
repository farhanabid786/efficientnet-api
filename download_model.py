"""
download_model.py
─────────────────
Downloads the EfficientNet-B4 checkpoint from Google Drive at startup.
Run ONCE before uvicorn starts (called by start.sh).

Why download at runtime instead of bundling in the repo?
  - Render free tier has a 512 MB slug (build artifact) size limit.
  - A 78 MB .pth pushed to Git quickly exceeds that + triggers LFS fees.
  - Downloading at container start keeps the repo clean and avoids limits.
"""

import os
import sys
import time

MODEL_PATH   = os.getenv("MODEL_PATH",     "models/best_model_v3.pth")
GDRIVE_ID    = os.getenv("MODEL_GDRIVE_ID", "1xigFpgBhZrlwRn4FWX8LZaf7_Rq5E7G3")


def download_from_gdrive(file_id: str, dest: str) -> None:
    """Download a Google Drive file using gdown (handles large-file cookies)."""
    try:
        import gdown
    except ImportError:
        print("[download] gdown not installed — run: pip install gdown")
        sys.exit(1)

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[download] Downloading model from Google Drive …")
    print(f"[download]   File ID : {file_id}")
    print(f"[download]   Dest    : {dest}")
    t0 = time.time()
    gdown.download(url, dest, quiet=False, fuzzy=True)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(dest) / 1e6
    print(f"[download] Done in {elapsed:.1f}s — {size_mb:.1f} MB")


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        print(f"[download] Model already exists ({size_mb:.1f} MB) — skipping download.")
    else:
        download_from_gdrive(GDRIVE_ID, MODEL_PATH)
