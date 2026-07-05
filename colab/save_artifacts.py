"""
Run this script in Google Colab after training to export model artifacts.

It expects the following variables to exist in the notebook/global namespace:
- sentiment_model: trained Keras model for sentiment (e.g. tuned_model or bilstm_s)
- category_model: trained Keras model for category (e.g. bilstm_c)
- tokenizer: fitted Keras Tokenizer
- le_sent: fitted LabelEncoder for sentiment
- le_cat: fitted LabelEncoder for category
- MAX_LEN, MAX_WORDS, EMBED_DIM: ints
- Optionally: history objects h_tuned, history objects for category, and test data
  to compute metrics.json

Usage in Colab:
    %run save_artifacts.py
"""

import json
import os
import zipfile
from pathlib import Path

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("/content/tokopedia_artifacts")
SENT_DIR = BASE_DIR / "sentiment_model"
CAT_DIR = BASE_DIR / "category_model"

for d in (SENT_DIR, CAT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Try to pull variables from the notebook/global namespace
try:
    _sentiment_model = globals().get("sentiment_model") or globals().get("tuned_model") or globals().get("bilstm_s")
    _category_model = globals().get("category_model") or globals().get("bilstm_c")
    _tokenizer = globals()["tokenizer"]
    _le_sent = globals()["le_sent"]
    _le_cat = globals()["le_cat"]
    _MAX_LEN = globals().get("MAX_LEN", 100)
    _MAX_WORDS = globals().get("MAX_WORDS", 20000)
    _EMBED_DIM = globals().get("EMBED_DIM", 128)
except KeyError as exc:
    raise RuntimeError(
        f"Required variable {exc} not found in the notebook environment. "
        "Make sure you have run all training cells before exporting."
    )


def _tokenizer_to_json(tokenizer):
    """Return tokenizer config as a JSON-serializable dict."""
    return tokenizer.to_json()


def _save_model_and_assets(model, out_dir, label_encoder, model_type, classes, metrics=None):
    """Save Keras model, tokenizer copy, label encoder, and config."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = out_dir / "model.keras"
    model.save(str(model_path))
    print(f"Saved model: {model_path}")

    # Tokenizer (shared between tasks in the notebook)
    tok_json = _tokenizer_to_json(_tokenizer)
    with open(out_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(tok_json, f, ensure_ascii=False, indent=2)
    print(f"Saved tokenizer: {out_dir / 'tokenizer.json'}")

    # Label encoder
    joblib.dump(label_encoder, out_dir / "label_encoder.pkl")
    print(f"Saved label encoder: {out_dir / 'label_encoder.pkl'}")

    # Config
    cfg = {
        "task": out_dir.name.replace("_model", ""),
        "model_type": model_type,
        "classes": list(classes),
        "max_len": int(_MAX_LEN),
        "max_words": int(_MAX_WORDS),
        "embed_dim": int(_EMBED_DIM),
        "vocab_size": min(_MAX_WORDS, len(_tokenizer.word_index)) + 1,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"Saved config: {out_dir / 'config.json'}")

    # Metrics (optional)
    if metrics:
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics: {out_dir / 'metrics.json'}")


# ---------------------------------------------------------------------------
# Sentiment artifacts
# ---------------------------------------------------------------------------
sent_metrics = None
if "X_te_s" in globals() and "y_te_s" in globals():
    try:
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        y_true = globals()["y_te_s"]
        X_te = globals()["X_te_s"]
        probs = _sentiment_model.predict(X_te, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        sent_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "classification_report": classification_report(
                y_true, y_pred, target_names=_le_sent.classes_, output_dict=True
            ),
        }
    except Exception as exc:
        print(f"Could not compute sentiment metrics: {exc}")

_save_model_and_assets(
    _sentiment_model,
    SENT_DIR,
    _le_sent,
    model_type="bilstm_tuned",
    classes=_le_sent.classes_,
    metrics=sent_metrics,
)

# ---------------------------------------------------------------------------
# Category artifacts
# ---------------------------------------------------------------------------
cat_metrics = None
if "X_te_c" in globals() and "y_te_c" in globals():
    try:
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        y_true = globals()["y_te_c"]
        X_te = globals()["X_te_c"]
        probs = _category_model.predict(X_te, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        cat_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "classification_report": classification_report(
                y_true, y_pred, target_names=_le_cat.classes_, output_dict=True
            ),
        }
    except Exception as exc:
        print(f"Could not compute category metrics: {exc}")

_save_model_and_assets(
    _category_model,
    CAT_DIR,
    _le_cat,
    model_type="bilstm",
    classes=_le_cat.classes_,
    metrics=cat_metrics,
)

# ---------------------------------------------------------------------------
# Zip and download
# ---------------------------------------------------------------------------
zip_path = Path("/content/tokopedia_artifacts.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(BASE_DIR):
        for file in files:
            fp = Path(root) / file
            zf.write(fp, arcname=fp.relative_to(BASE_DIR.parent))

print(f"\nArtifacts zipped: {zip_path}")
print("Downloading...")
try:
    from google.colab import files

    files.download(str(zip_path))
except Exception as exc:
    print(f"Could not trigger download automatically: {exc}")
    print(f"Please download {zip_path} manually from the Files panel on the left.")
