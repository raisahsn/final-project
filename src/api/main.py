"""FastAPI application serving Tokopedia review predictions."""

import json
from pathlib import Path
from typing import List

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from tokopedia_ml import config
from tokopedia_ml.inference import get_predictor

from .database import (
    Prediction,
    get_db,
    init_db,
    save_batch_predictions,
    save_prediction,
)
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    BatchItem,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    PredictionsListResponse,
    PredictionRecord,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables on startup."""
    init_db()
    yield


app = FastAPI(
    title="Tokopedia Review Prediction API",
    description="REST API for sentiment and category classification of Tokopedia product reviews.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_model_info(task: str) -> ModelInfo:
    """Load model metadata from config.json if available."""
    model_dir = Path(config.DEFAULT_MODELS[task]["dir"])
    cfg_path = model_dir / "config.json"
    loaded = (model_dir / "model.keras").exists() and cfg_path.exists()
    if loaded and cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return ModelInfo(
            task=task,
            model_type=cfg.get("model_type", "unknown"),
            classes=cfg.get("classes", []),
            max_len=cfg.get("max_len", config.MAX_LEN),
            loaded=True,
        )
    return ModelInfo(
        task=task,
        model_type="unknown",
        classes=config.DEFAULT_MODELS[task].get("classes", []),
        max_len=config.MAX_LEN,
        loaded=False,
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return API health status and model availability."""
    return HealthResponse(
        status="ok",
        models=[
            _load_model_info(config.TASK_SENTIMENT),
            _load_model_info(config.TASK_CATEGORY),
        ],
    )


@app.get("/models", response_model=List[ModelInfo])
def list_models() -> List[ModelInfo]:
    """List loaded model metadata."""
    return [
        _load_model_info(config.TASK_SENTIMENT),
        _load_model_info(config.TASK_CATEGORY),
    ]


@app.post("/predict", response_model=PredictResponse)
def predict_single(
    request: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """Predict sentiment and category for a single review text."""
    try:
        result = get_predictor().predict_text(request.review_text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    save_prediction(db, result)
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    request: BatchPredictRequest,
    db: Session = Depends(get_db),
) -> BatchPredictResponse:
    """Predict sentiment and category for a batch of review texts."""
    try:
        results = get_predictor().predict_batch(request.texts)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    save_batch_predictions(db, results)
    return BatchPredictResponse(
        count=len(results),
        results=[BatchItem(**r) for r in results],
    )


@app.get("/predictions", response_model=PredictionsListResponse)
def get_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> PredictionsListResponse:
    """Retrieve stored predictions with pagination."""
    total = db.query(Prediction).count()
    offset = (page - 1) * page_size
    items = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return PredictionsListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[PredictionRecord.model_validate(item) for item in items],
    )
