"""Inference helpers for sentiment and category classification."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from . import config
from .preprocessing import (
    clean_text,
    detect_offensive,
    find_offensive_words,
    load_label_encoder,
    load_tokenizer,
    texts_to_padded,
    validate_input_text,
)

logger = logging.getLogger(__name__)

SENTIMENT_CONFIDENCE_THRESHOLD = 0.6


try:
    import tf_keras as keras_backend
except ImportError:
    import tensorflow.keras as keras_backend

    logger.warning(
        "tf_keras not installed; falling back to tensorflow.keras for loading."
    )


class ModelLoadError(Exception):
    """Raised when a model artifact cannot be loaded."""

    def __init__(self, task: str, reason: str):
        self.task = task
        self.reason = reason
        super().__init__(f"Failed to load model for '{task}': {reason}")


def _model_dir(task: str) -> Path:
    """Return the artifact directory for a given task."""
    return Path(config.DEFAULT_MODELS[task]["dir"])


def load_artifacts(task: str):
    """Load model, tokenizer, label encoder, and config for a task.

    Args:
        task: either config.TASK_SENTIMENT or config.TASK_CATEGORY.

    Returns:
        tuple: (keras.Model, Tokenizer, LabelEncoder, dict config)

    Raises:
        ModelLoadError: if any artifact is missing or cannot be loaded.
    """
    model_dir = _model_dir(task)
    logger.info("Loading artifacts for task='%s' from %s", task, model_dir)

    if not model_dir.exists():
        raise ModelLoadError(
            task,
            f"Model directory not found at {model_dir}. "
            "Please export artifacts from Google Colab using colab/save_artifacts.py.",
        )

    required_files = {
        "model": model_dir / "model.keras",
        "tokenizer": model_dir / "tokenizer.json",
        "label_encoder": model_dir / "label_encoder.pkl",
        "config": model_dir / "config.json",
    }

    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        raise ModelLoadError(
            task,
            f"Missing artifact files: {missing}. Expected them in {model_dir}.",
        )

    try:
        with open(required_files["config"], "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as exc:
        raise ModelLoadError(task, f"Failed to read config.json: {exc}") from exc

    try:
        model = keras_backend.models.load_model(str(required_files["model"]))
        logger.info("Loaded Keras model for task='%s'", task)
    except Exception as first_exc:
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(str(required_files["model"]))
            logger.info("Loaded Keras model for task='%s' using tensorflow.keras", task)
        except Exception as second_exc:
            raise ModelLoadError(
                task,
                f"Failed to load model.keras. "
                f"Common causes: Keras/TensorFlow version mismatch between training and inference, "
                f"or corrupted file. If the model was saved in Google Colab with tf_keras, "
                f"install tf-keras==2.16.0. "
                f"tf_keras error: {first_exc}; tensorflow.keras error: {second_exc}",
            ) from second_exc

    try:
        tokenizer = load_tokenizer(required_files["tokenizer"])
    except Exception as exc:
        raise ModelLoadError(task, f"Failed to load tokenizer.json: {exc}") from exc

    try:
        label_encoder = load_label_encoder(required_files["label_encoder"])
    except Exception as exc:
        raise ModelLoadError(task, f"Failed to load label_encoder.pkl: {exc}") from exc

    return model, tokenizer, label_encoder, cfg


class Predictor:
    """Loads both sentiment and category models and exposes predict methods."""

    def __init__(self):
        self._errors: Dict[str, str] = {}
        self.sent_model = None
        self.sent_tokenizer = None
        self.sent_le = None
        self.sent_cfg = None
        self.cat_model = None
        self.cat_tokenizer = None
        self.cat_le = None
        self.cat_cfg = None

        try:
            (
                self.sent_model,
                self.sent_tokenizer,
                self.sent_le,
                self.sent_cfg,
            ) = load_artifacts(config.TASK_SENTIMENT)
        except ModelLoadError as exc:
            logger.error(str(exc))
            self._errors[config.TASK_SENTIMENT] = exc.reason

        try:
            (
                self.cat_model,
                self.cat_tokenizer,
                self.cat_le,
                self.cat_cfg,
            ) = load_artifacts(config.TASK_CATEGORY)
        except ModelLoadError as exc:
            logger.error(str(exc))
            self._errors[config.TASK_CATEGORY] = exc.reason

        if self._errors:
            loaded = [
                t
                for t in (config.TASK_SENTIMENT, config.TASK_CATEGORY)
                if t not in self._errors
            ]
            logger.warning(
                "Predictor initialized with errors. Loaded: %s, Failed: %s",
                loaded,
                list(self._errors.keys()),
            )
        else:
            logger.info("Predictor initialized successfully for all tasks.")

    @property
    def errors(self) -> Dict[str, str]:
        """Return mapping of task -> error message for failed loads."""
        return self._errors.copy()

    def is_ready(self) -> bool:
        """Return True if both models loaded successfully."""
        return (
            self.sent_model is not None
            and self.cat_model is not None
            and not self._errors
        )

    def _ensure_ready(self) -> None:
        """Raise an informative error if predictor is not fully ready."""
        if not self.is_ready():
            details = "; ".join(
                f"{task}: {reason}" for task, reason in self._errors.items()
            )
            raise ModelLoadError(
                "all",
                f"Models are not fully loaded. Details: {details}",
            )

    def predict_text(self, text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """Predict sentiment and category for a single review."""
        self._ensure_ready()
        validated = validate_input_text(text)
        cleaned = clean_text(validated)
        if not cleaned:
            raise ValueError("Text became empty after cleaning; cannot predict.")

        offensive = detect_offensive(validated)
        offensive_words = find_offensive_words(validated) if offensive else []

        if offensive:
            logger.info(
                "Offensive language detected in review; applying rule-based negative sentiment."
            )
            sentiment = {
                "label": "negative",
                "confidence": 1.0,
                "probabilities": {"negative": 1.0, "neutral": 0.0, "positive": 0.0},
            }
        else:
            sentiment = self._predict_one(
                cleaned,
                self.sent_model,
                self.sent_tokenizer,
                self.sent_le,
                self.sent_cfg["max_len"],
            )

        category = self._predict_one(
            cleaned,
            self.cat_model,
            self.cat_tokenizer,
            self.cat_le,
            self.cat_cfg["max_len"],
        )

        low_confidence = sentiment["confidence"] < SENTIMENT_CONFIDENCE_THRESHOLD
        rule_based_sentiment = offensive

        note = ""
        if offensive:
            note = "Sentiment determined by offensive-language filter."
        elif low_confidence:
            note = f"Low sentiment confidence ({sentiment['confidence']:.1%}); review manually if needed."

        return {
            "review_text": validated,
            "cleaned_text": cleaned,
            "sentiment": {
                "label": sentiment["label"],
                "confidence": round(sentiment["confidence"], 4),
                "probabilities": {
                    k: round(v, 4) for k, v in sentiment["probabilities"].items()
                },
            },
            "category": {
                "label": category["label"],
                "confidence": round(category["confidence"], 4),
                "probabilities": {
                    k: round(v, 4) for k, v in category["probabilities"].items()
                },
            },
            "flags": {
                "offensive": offensive,
                "offensive_words": offensive_words,
                "low_confidence": low_confidence,
                "rule_based_sentiment": rule_based_sentiment,
                "note": note,
            },
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment and category for a list of reviews."""
        self._ensure_ready()
        cleaned = []
        offensive_flags = []
        for t in texts:
            validated = validate_input_text(t)
            ct = clean_text(validated)
            offensive = detect_offensive(validated)
            cleaned.append((validated, ct, offensive))
            offensive_flags.append(offensive)

        # Only run model on non-offensive texts for sentiment
        non_offensive_indices = [
            i for i, (_, _, offensive) in enumerate(cleaned) if not offensive
        ]
        non_offensive_cleaned = [cleaned[i][1] for i in non_offensive_indices]

        sent_probs_all = [None] * len(texts)
        if non_offensive_cleaned:
            sent_probs = self._predict_batch(
                non_offensive_cleaned,
                self.sent_model,
                self.sent_tokenizer,
                self.sent_cfg["max_len"],
            )
            for idx, prob in zip(non_offensive_indices, sent_probs):
                sent_probs_all[idx] = prob

        # Category model runs on all texts
        cat_probs = self._predict_batch(
            [c for _, c, _ in cleaned],
            self.cat_model,
            self.cat_tokenizer,
            self.cat_cfg["max_len"],
        )

        results = []
        for (original, _, offensive), s_probs, c_probs in zip(
            cleaned, sent_probs_all, cat_probs
        ):
            if offensive:
                sentiment_label = "negative"
                sentiment_confidence = 1.0
            else:
                sentiment_label = self._top_label(s_probs, self.sent_le)
                sentiment_confidence = round(float(max(s_probs)), 4)

            results.append(
                {
                    "review_text": original,
                    "sentiment_label": sentiment_label,
                    "sentiment_confidence": sentiment_confidence,
                    "category_label": self._top_label(c_probs, self.cat_le),
                    "category_confidence": round(float(max(c_probs)), 4),
                    "offensive": offensive,
                    "offensive_words": (
                        find_offensive_words(original) if offensive else []
                    ),
                    "low_confidence": sentiment_confidence
                    < SENTIMENT_CONFIDENCE_THRESHOLD,
                }
            )
        return results

    @staticmethod
    def _predict_one(
        cleaned: str,
        model,
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
        model,
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
_predictor_init_error: Optional[str] = None


def get_predictor() -> Predictor:
    """Return a cached Predictor instance."""
    global _predictor, _predictor_init_error
    if _predictor is None:
        try:
            _predictor = Predictor()
        except Exception as exc:
            _predictor_init_error = str(exc)
            logger.exception("Failed to initialize predictor")
            raise
    return _predictor


def get_predictor_status() -> Tuple[bool, Dict[str, str]]:
    """Return (is_ready, errors) without raising on failure."""
    global _predictor, _predictor_init_error
    if _predictor is not None:
        return _predictor.is_ready(), _predictor.errors
    try:
        predictor = get_predictor()
        return predictor.is_ready(), predictor.errors
    except Exception as exc:
        return False, {"init": str(exc) or _predictor_init_error or "unknown error"}


def predict_text(text: str) -> Dict:
    """Convenience function: predict a single text."""
    return get_predictor().predict_text(text)


def predict_batch(texts: List[str]) -> List[Dict]:
    """Convenience function: predict a batch of texts."""
    return get_predictor().predict_batch(texts)
