"""Configuration for Tokopedia ML inference and training."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

# Default hyperparameters matching the notebook
MAX_WORDS = 20_000
MAX_LEN = 100
EMBED_DIM = 128
OOV_TOKEN = "<OOV>"

TASK_SENTIMENT = "sentiment"
TASK_CATEGORY = "category"

DEFAULT_MODELS = {
    TASK_SENTIMENT: {
        "dir": MODELS_DIR / "sentiment_model",
        "type": "bilstm_tuned",
        "classes": ["negative", "neutral", "positive"],
    },
    TASK_CATEGORY: {
        "dir": MODELS_DIR / "category_model",
        "type": "bilstm",
        "classes": ["produk", "produk_dan_pengiriman", "pengiriman", "umum"],
    },
}

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'predictions.db'}")

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Streamlit
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
