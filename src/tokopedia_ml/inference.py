"""Inference helpers for sentiment and category classification."""

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf

from . import config
from .preprocessing import (
    clean_text,
    load_label_encoder,
    load_tokenizer,
    texts_to_padded,
    validate_input_text,
)


def _model_dir(task: str) -> Path:
    """Return the artifact directory for a given task."""
    return Path(config.DEFAULT_MODELS[task]["dir"])


def load_artifacts(task: str):
    """Load model, tokenizer, label encoder, and config for a task.

    Args:
        task: either config.TASK_SENTIMENT or config.TASK_CATEGORY.

    Returns:
        tuple: (keras.Model, Tokenizer, LabelEncoder, dict config)
    """
    model_dir = _model_dir(task)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model artifacts for '{task}' not found at {model_dir}. "
            "Please export them from Google Colab using colab/save_artifacts.py."
        )

    cfg_path = model_dir / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = tf.keras.models.load_model(str(model_dir / "model.keras"))
    tokenizer = load_tokenizer(model_dir / "tokenizer.json")
    label_encoder = load_label_encoder(model_dir / "label_encoder.pkl")

    return model, tokenizer, label_encoder, cfg


class Predictor:
    """Loads both sentiment and category models and exposes predict methods."""

    def __init__(self):
        self.sent_model, self.sent_tokenizer, self.sent_le, self.sent_cfg = (
            load_artifacts(config.TASK_SENTIMENT)
        )
        self.cat_model, self.cat_tokenizer, self.cat_le, self.cat_cfg = load_artifacts(
            config.TASK_CATEGORY
        )

    def predict_text(self, text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """Predict sentiment and category for a single review."""
        validated = validate_input_text(text)
        cleaned = clean_text(validated)
        if not cleaned:
            raise ValueError("Text became empty after cleaning; cannot predict.")

        sent = self._predict_one(
            cleaned,
            self.sent_model,
            self.sent_tokenizer,
            self.sent_le,
            self.sent_cfg["max_len"],
        )
        cat = self._predict_one(
            cleaned,
            self.cat_model,
            self.cat_tokenizer,
            self.cat_le,
            self.cat_cfg["max_len"],
        )

        return {
            "review_text": validated,
            "cleaned_text": cleaned,
            "sentiment": {
                "label": sent["label"],
                "confidence": round(sent["confidence"], 4),
                "probabilities": {
                    k: round(v, 4) for k, v in sent["probabilities"].items()
                },
            },
            "category": {
                "label": cat["label"],
                "confidence": round(cat["confidence"], 4),
                "probabilities": {
                    k: round(v, 4) for k, v in cat["probabilities"].items()
                },
            },
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment and category for a list of reviews."""
        cleaned = []
        for t in texts:
            validated = validate_input_text(t)
            ct = clean_text(validated)
            cleaned.append((validated, ct))

        sent_probs = self._predict_batch(
            [c for _, c in cleaned],
            self.sent_model,
            self.sent_tokenizer,
            self.sent_cfg["max_len"],
        )
        cat_probs = self._predict_batch(
            [c for _, c in cleaned],
            self.cat_model,
            self.cat_tokenizer,
            self.cat_cfg["max_len"],
        )

        results = []
        for (original, _), s_probs, c_probs in zip(cleaned, sent_probs, cat_probs):
            results.append(
                {
                    "review_text": original,
                    "sentiment_label": self._top_label(s_probs, self.sent_le),
                    "sentiment_confidence": round(float(max(s_probs)), 4),
                    "category_label": self._top_label(c_probs, self.cat_le),
                    "category_confidence": round(float(max(c_probs)), 4),
                }
            )
        return results

    @staticmethod
    def _predict_one(
        cleaned: str,
        model: tf.keras.Model,
        tokenizer,
        label_encoder,
        max_len: int,
    ) -> Dict:
        padded = texts_to_padded([cleaned], tokenizer, max_len)
        probs = model.predict(padded, verbose=0)[0]
        return Predictor._format_probs(probs, label_encoder)

    @staticmethod
    def _predict_batch(
        cleaned_texts: List[str],
        model: tf.keras.Model,
        tokenizer,
        max_len: int,
    ) -> np.ndarray:
        padded = texts_to_padded(cleaned_texts, tokenizer, max_len)
        return model.predict(padded, verbose=0)

    @staticmethod
    def _format_probs(probs: np.ndarray, label_encoder) -> Dict:
        classes = list(label_encoder.classes_)
        idx = int(np.argmax(probs))
        return {
            "label": classes[idx],
            "confidence": float(probs[idx]),
            "probabilities": {cls: float(probs[i]) for i, cls in enumerate(classes)},
        }

    @staticmethod
    def _top_label(probs: np.ndarray, label_encoder) -> str:
        return str(label_encoder.classes_[int(np.argmax(probs))])


# Module-level singleton for FastAPI/Streamlit reuse
_predictor: Union[Predictor, None] = None


def get_predictor() -> Predictor:
    """Return a cached Predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def predict_text(text: str) -> Dict:
    """Convenience function: predict a single text."""
    return get_predictor().predict_text(text)


def predict_batch(texts: List[str]) -> List[Dict]:
    """Convenience function: predict a batch of texts."""
    return get_predictor().predict_batch(texts)
