#!/usr/bin/env python3
"""Download model artifacts (zip) at container startup.

Behaviour:
- If MODELS_URL is not set, do nothing (assumes models/ already present in image).
- If models already exist, skip download.
- Otherwise download the zip and extract it into MODELS_DIR.

Expected zip layout (root-level):
    sentiment_model/
    category_model/

Create it from the notebook output like:
    cd tokopedia_artifacts && zip -r ../models-deploy.zip sentiment_model category_model

Env vars:
    MODELS_URL  -> public URL to the zip (S3/GCS/R2/Hugging Face direct link, etc.)
    MODELS_DIR  -> target directory (default: "models")
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path


MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
URL = os.environ.get("MODELS_URL", "").strip()


def _models_ready() -> bool:
    return (MODELS_DIR / "sentiment_model" / "model.keras").exists() and (
        MODELS_DIR / "category_model" / "model.keras"
    ).exists()


def main() -> int:
    if not URL:
        print("[download_models] MODELS_URL not set; skipping download.")
        return 0

    if _models_ready():
        print("[download_models] Models already present; skipping download.")
        return 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/models.zip")

    print(f"[download_models] Downloading models from {URL} ...")
    try:
        urllib.request.urlretrieve(URL, tmp)
    except Exception as exc:  # noqa: BLE001
        print(f"[download_models] Failed to download: {exc}")
        return 1

    print("[download_models] Extracting zip ...")
    try:
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(MODELS_DIR)
    except Exception as exc:  # noqa: BLE001
        print(f"[download_models] Failed to extract: {exc}")
        return 1

    if _models_ready():
        print("[download_models] Models ready ✅")
        return 0

    print(
        "[download_models] Extract finished but models not found at expected paths. "
        "Make sure the zip contains sentiment_model/ and category_model/ at the root."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
