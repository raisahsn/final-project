"""Pydantic schemas for the FastAPI application."""

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    review_text: str = Field(
        ..., min_length=1, max_length=2000, description="Raw Indonesian review text"
    )


class ProbabilityMap(BaseModel):
    probabilities: Dict[str, float]
    label: str
    confidence: float


class PredictResponse(BaseModel):
    review_text: str
    cleaned_text: str
    sentiment: ProbabilityMap
    category: ProbabilityMap


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ..., min_length=1, max_length=100, description="List of review texts"
    )


class BatchItem(BaseModel):
    review_text: str
    sentiment_label: str
    sentiment_confidence: float
    category_label: str
    category_confidence: float


class BatchPredictResponse(BaseModel):
    count: int
    results: List[BatchItem]


class PredictionRecord(BaseModel):
    id: int
    review_text: str
    sentiment_label: str
    sentiment_confidence: float
    category_label: str
    category_confidence: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ModelInfo(BaseModel):
    task: str
    model_type: str
    classes: List[str]
    max_len: int
    loaded: bool

    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(BaseModel):
    status: str
    models: List[ModelInfo]


class PredictionsListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[PredictionRecord]
