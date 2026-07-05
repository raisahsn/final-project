"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.database import init_db
from api.main import app


@pytest.fixture(scope="module")
def client():
    init_db()
    return TestClient(app)


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert len(data["models"]) == 2


def test_predict_single(client):
    resp = client.post(
        "/predict", json={"review_text": "barang bagus pengiriman cepat"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "sentiment" in data
    assert "category" in data
    assert data["review_text"] == "barang bagus pengiriman cepat"


def test_predict_batch(client):
    resp = client.post(
        "/predict/batch",
        json={"texts": ["bagus", "jelek"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2


def test_predictions_list(client):
    client.post("/predict", json={"review_text": "bagus"})
    resp = client.get("/predictions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    assert "items" in data


def test_predict_empty_text_returns_400(client):
    resp = client.post("/predict", json={"review_text": ""})
    assert resp.status_code == 422  # Pydantic validation
