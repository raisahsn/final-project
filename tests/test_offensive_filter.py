"""Tests for offensive language filter and confidence threshold."""

from fastapi.testclient import TestClient

from api.database import init_db
from api.main import app


def test_offensive_text_returns_negative_sentiment():
    init_db()
    client = TestClient(app)
    resp = client.post("/predict", json={"review_text": "barang jelek anjing"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sentiment"]["label"] == "negative"
    assert data["sentiment"]["confidence"] == 1.0
    assert data["flags"]["offensive"] is True
    assert "anjing" in data["flags"]["offensive_words"]
    assert data["flags"]["rule_based_sentiment"] is True


def test_clean_text_uses_model_prediction():
    init_db()
    client = TestClient(app)
    resp = client.post(
        "/predict", json={"review_text": "barang bagus pengiriman cepat"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["flags"]["offensive"] is False
    assert data["flags"]["rule_based_sentiment"] is False


def test_batch_offensive_detection():
    init_db()
    client = TestClient(app)
    resp = client.post(
        "/predict/batch",
        json={"texts": ["barang jelek anjing", "barang bagus"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert data["results"][0]["sentiment_label"] == "negative"
    assert data["results"][0]["offensive"] is True
    assert data["results"][1]["offensive"] is False


def test_offensive_words_helper():
    from tokopedia_ml.preprocessing import detect_offensive, find_offensive_words

    assert detect_offensive("barang jelek anjing") is True
    assert "anjing" in find_offensive_words("barang jelek anjing")
    assert detect_offensive("barang bagus") is False
