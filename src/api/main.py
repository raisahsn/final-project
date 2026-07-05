"""FastAPI application serving Tokopedia review predictions."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from tokopedia_ml import config
from tokopedia_ml.inference import (
    ModelLoadError,
    get_predictor,
    get_predictor_status,
)

from .database import (
    get_db,
    init_db,
    query_predictions,
    save_batch_predictions,
    save_prediction,
)
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    BatchItem,
    Flags,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    PredictionsListResponse,
    PredictionRecord,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables and attempt to load models on startup."""
    init_db()
    ready, errors = get_predictor_status()
    if ready:
        logger.info("All models loaded successfully at startup.")
    else:
        logger.warning("Models not fully loaded at startup. Errors: %s", errors)
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


def _load_model_info(task: str, errors: Dict[str, str]) -> ModelInfo:
    """Load model metadata from config.json if available."""
    model_dir = Path(config.DEFAULT_MODELS[task]["dir"])
    cfg_path = model_dir / "config.json"
    loaded = (model_dir / "model.keras").exists() and cfg_path.exists()
    error_msg = errors.get(task, "")

    if loaded and cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return ModelInfo(
                task=task,
                model_type=cfg.get("model_type", "unknown"),
                classes=cfg.get("classes", []),
                max_len=cfg.get("max_len", config.MAX_LEN),
                loaded=not error_msg,
                error=error_msg,
            )
        except Exception as exc:
            logger.warning("Could not read config.json for %s: %s", task, exc)

    return ModelInfo(
        task=task,
        model_type="unknown",
        classes=config.DEFAULT_MODELS[task].get("classes", []),
        max_len=config.MAX_LEN,
        loaded=False,
        error=error_msg or f"Model artifacts not found at {model_dir}",
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return API health status and detailed model availability/errors."""
    ready, errors = get_predictor_status()
    return HealthResponse(
        status="ok",
        ready=ready,
        models=[
            _load_model_info(config.TASK_SENTIMENT, errors),
            _load_model_info(config.TASK_CATEGORY, errors),
        ],
        errors=errors,
    )


@app.get("/models", response_model=List[ModelInfo])
def list_models() -> List[ModelInfo]:
    """List loaded model metadata with error details."""
    _, errors = get_predictor_status()
    return [
        _load_model_info(config.TASK_SENTIMENT, errors),
        _load_model_info(config.TASK_CATEGORY, errors),
    ]


def _handle_prediction_error(exc: Exception) -> None:
    """Convert prediction exceptions to informative HTTP errors."""
    if isinstance(exc, ModelLoadError):
        logger.error("Model not loaded: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not loaded",
                "message": str(exc),
                "task": exc.task,
                "hint": "Verify model artifacts in models/ match the TensorFlow/Keras version used for inference.",
            },
        ) from exc
    if isinstance(exc, FileNotFoundError):
        logger.error("File not found: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model artifacts missing",
                "message": str(exc),
                "hint": "Export artifacts from Google Colab and extract them to models/.",
            },
        ) from exc
    if isinstance(exc, ValueError):
        logger.warning("Bad request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.exception("Unexpected prediction error")
    raise HTTPException(
        status_code=500,
        detail={
            "error": "Internal prediction error",
            "message": str(exc),
            "hint": "Check server logs for details.",
        },
    ) from exc


@app.post("/predict", response_model=PredictResponse)
def predict_single(
    request: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """Predict sentiment and category for a single review text."""
    try:
        result = get_predictor().predict_text(request.review_text)
    except Exception as exc:
        _handle_prediction_error(exc)

    saved = save_prediction(db, result)
    if saved is None:
        logger.warning("Prediction computed but not persisted due to database issue.")
    return PredictResponse(
        review_text=result["review_text"],
        cleaned_text=result["cleaned_text"],
        sentiment=result["sentiment"],
        category=result["category"],
        flags=Flags(**result["flags"]),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    request: BatchPredictRequest,
    db: Session = Depends(get_db),
) -> BatchPredictResponse:
    """Predict sentiment and category for a batch of review texts."""
    try:
        results = get_predictor().predict_batch(request.texts)
    except Exception as exc:
        _handle_prediction_error(exc)

    saved_records = save_batch_predictions(db, results)
    if not saved_records:
        logger.warning("Batch predictions computed but not persisted due to database issue.")
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
    offset = (page - 1) * page_size
    total, items = query_predictions(db, skip=offset, limit=page_size)
    return PredictionsListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[PredictionRecord.model_validate(item) for item in items],
    )
