"""Shared pytest fixtures."""

import json
import shutil
from pathlib import Path

import joblib
import pytest
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers

from tokopedia_ml import config
from tokopedia_ml.preprocessing import save_tokenizer


@pytest.fixture(scope="session", autouse=True)
def create_dummy_artifacts():
    """Create tiny dummy model artifacts so tests can run without real training."""
    created_dirs = []

    # shared tokenizer
    tokenizer = Tokenizer(num_words=config.MAX_WORDS, oov_token=config.OOV_TOKEN)
    tokenizer.fit_on_texts(["bagus cepat jelek lambat produk pengiriman"])

    for task, info in config.DEFAULT_MODELS.items():
        model_dir = Path(info["dir"])
        if (model_dir / "model.keras").exists():
            continue  # real artifacts present, do not overwrite

        model_dir.mkdir(parents=True, exist_ok=True)
        created_dirs.append(model_dir)

        # tiny model
        inp = keras.Input(shape=(config.MAX_LEN,))
        x = layers.Embedding(config.MAX_WORDS, 8)(inp)
        x = layers.GlobalMaxPooling1D()(x)
        out = layers.Dense(len(info["classes"]), activation="softmax")(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.save(str(model_dir / "model.keras"))

        # tokenizer (real Keras Tokenizer JSON)
        save_tokenizer(tokenizer, model_dir / "tokenizer.json")

        # label encoder
        le = LabelEncoder()
        le.fit(info["classes"])
        joblib.dump(le, model_dir / "label_encoder.pkl")

        # config
        cfg = {
            "task": task,
            "model_type": info["type"],
            "classes": list(info["classes"]),
            "max_len": config.MAX_LEN,
            "max_words": config.MAX_WORDS,
            "embed_dim": config.EMBED_DIM,
            "vocab_size": config.MAX_WORDS,
        }
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    yield

    # cleanup only directories we created
    for d in created_dirs:
        if d.exists():
            shutil.rmtree(d)
