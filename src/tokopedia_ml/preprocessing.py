"""Text preprocessing utilities for Tokopedia review classification."""

import json
import re
import unicodedata
from pathlib import Path
from typing import List, Union

import joblib
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    _stemmer = StemmerFactory().create_stemmer()
    _HAS_SASTRAWI = True
except Exception:  # pragma: no cover - Sastrawi is optional in some envs
    _stemmer = None
    _HAS_SASTRAWI = False


STOPWORDS = set(
    [
        "yang",
        "dan",
        "di",
        "ke",
        "dari",
        "ini",
        "itu",
        "dengan",
        "untuk",
        "pada",
        "adalah",
        "ada",
        "juga",
        "sudah",
        "saya",
        "aku",
        "kamu",
        "dia",
        "mereka",
        "kita",
        "kami",
        "bisa",
        "akan",
        "tidak",
        "ya",
        "ga",
        "gak",
        "nggak",
        "udah",
        "belum",
        "nya",
        "lah",
        "tapi",
        "atau",
        "karena",
        "jadi",
        "kalau",
        "sama",
        "seperti",
        "lebih",
        "sangat",
        "banget",
        "sekali",
        "pun",
        "sih",
        "deh",
        "dong",
        "nih",
        "yg",
        "dgn",
        "utk",
        "krn",
        "sdh",
        "blm",
        "tp",
        "dr",
        "pd",
        "bgt",
        "sy",
        "gue",
        "lo",
        "lu",
        "emg",
        "msh",
        "jg",
        "jga",
        "aja",
        "doang",
        "lg",
        "lagi",
    ]
)


def clean_text(text: str) -> str:
    """Clean and normalize Indonesian review text.

    Mirrors the pipeline from the research notebook:
    lowercasing, NFKD normalization, URL/HTML removal, non-alpha removal,
    stopword removal, and (when available) Sastrawi stemming.
    """
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    if _HAS_SASTRAWI:
        tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


def validate_input_text(text: Union[str, None], max_chars: int = 2000) -> str:
    """Validate and normalize a single input string."""
    if text is None:
        raise ValueError("Input text cannot be None.")
    text = str(text).strip()
    if not text:
        raise ValueError("Input text cannot be empty.")
    if len(text) > max_chars:
        raise ValueError(
            f"Input text exceeds maximum length of {max_chars} characters."
        )
    return text


def save_tokenizer(tokenizer: Tokenizer, path: Union[str, Path]) -> None:
    """Save a Keras Tokenizer to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.to_json(), f, ensure_ascii=False, indent=2)


def load_tokenizer(path: Union[str, Path]) -> Tokenizer:
    """Load a Keras Tokenizer from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data may be a JSON string or a dict depending on how it was saved.
    if isinstance(data, dict):
        data = json.dumps(data, ensure_ascii=False)
    return tokenizer_from_json(data)


def save_label_encoder(encoder, path: Union[str, Path]) -> None:
    """Save a scikit-learn LabelEncoder with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, path)


def load_label_encoder(path: Union[str, Path]):
    """Load a scikit-learn LabelEncoder with joblib."""
    return joblib.load(path)


def texts_to_padded(
    texts: List[str],
    tokenizer: Tokenizer,
    max_len: int,
):
    """Transform raw cleaned texts into padded sequences."""
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
