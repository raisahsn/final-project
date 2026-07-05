"""Tests for model loading error handling."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.database import init_db
from api.main import app
from tokopedia_ml import config
from tokopedia_ml.inference import ModelLoadError, Predictor, load_artifacts


def test_load_artifacts_missing_directory(tmp_path):
    task = config.TASK_SENTIMENT
    original_dir = config.DEFAULT_MODELS[task]["dir"]
    config.DEFAULT_MODELS[task]["dir"] = tmp_path / "does_not_exist"
    try:
        with pytest.raises(ModelLoadError) as exc_info:
            load_artifacts(task)
        assert "Model directory not found" in str(exc_info.value)
    finally:
        config.DEFAULT_MODELS[task]["dir"] = original_dir


def test_load_artifacts_missing_files(tmp_path):
    task = config.TASK_SENTIMENT
    original_dir = config.DEFAULT_MODELS[task]["dir"]
    config.DEFAULT_MODELS[task]["dir"] = tmp_path
    try:
        with pytest.raises(ModelLoadError) as exc_info:
            load_artifacts(task)
        assert "Missing artifact files" in str(exc_info.value)
    finally:
        config.DEFAULT_MODELS[task]["dir"] = original_dir


def test_predictor_stores_error_for_missing_model(tmp_path):
    task = config.TASK_SENTIMENT
    original_dir = config.DEFAULT_MODELS[task]["dir"]
    config.DEFAULT_MODELS[task]["dir"] = tmp_path
    try:
        predictor = Predictor()
        assert not predictor.is_ready()
        assert task in predictor.errors
        with pytest.raises(ModelLoadError):
            predictor.predict_text("bagus")
    finally:
        config.DEFAULT_MODELS[task]["dir"] = original_dir


def test_api_returns_503_when_model_not_loaded():
    init_db()
    bad_predictor = MagicMock(spec=Predictor)
    bad_predictor.is_ready.return_value = False
    bad_predictor.errors = {
        config.TASK_SENTIMENT: "Missing artifact files: ['model']",
        config.TASK_CATEGORY: "Missing artifact files: ['model']",
    }
    bad_predictor.predict_text.side_effect = ModelLoadError(
        "all", "Models are not fully loaded."
    )

    client = TestClient(app)
    with patch("api.main.get_predictor", return_value=bad_predictor):
        response = client.post("/predict", json={"review_text": "bagus"})

    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"] == "Model not loaded"


def test_health_returns_errors_when_models_fail():
    init_db()
    errors = {
        config.TASK_SENTIMENT: "Keras version mismatch",
    }

    client = TestClient(app)
    with patch("api.main.get_predictor_status", return_value=(False, errors)):
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is False
    assert data["errors"] == errors
    sentiment_model = next(
        m for m in data["models"] if m["task"] == config.TASK_SENTIMENT
    )
    assert sentiment_model["loaded"] is False
    assert "Keras version mismatch" in sentiment_model["error"]
