"""Database layer for persisting predictions."""

import logging
from datetime import datetime
from typing import Generator

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from tokopedia_ml import config

logger = logging.getLogger(__name__)

Base = declarative_base()


class Prediction(Base):
    """SQLAlchemy model for a stored prediction."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    review_text = Column(String(2000), nullable=False)
    sentiment_label = Column(String(50), nullable=False)
    sentiment_confidence = Column(Float, nullable=False)
    category_label = Column(String(50), nullable=False)
    category_confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


engine = create_engine(
    config.DATABASE_URL,
    connect_args=(
        {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}
    ),
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable WAL mode and reasonable durability settings for SQLite."""
    if config.DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA mmap_size=30000000000")
        finally:
            cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create tables if they do not exist."""
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as exc:
        logger.error("Failed to initialize database: %s", exc)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session for FastAPI dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(db: Session, result: dict) -> Prediction | None:
    """Persist a single prediction result.

    Returns the saved record or None if persistence failed. Prediction failures
    should not break the API response.
    """
    record = Prediction(
        review_text=result["review_text"],
        sentiment_label=result["sentiment"]["label"],
        sentiment_confidence=result["sentiment"]["confidence"],
        category_label=result["category"]["label"],
        category_confidence=result["category"]["confidence"],
    )
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    except SQLAlchemyError as exc:
        db.rollback()
        logger.warning("Could not save prediction: %s", exc)
        return None


def save_batch_predictions(db: Session, results: list) -> list:
    """Persist a batch of prediction results.

    Returns the list of successfully persisted records; failed items are skipped
    so the overall batch response can still be returned.
    """
    records = []
    for r in results:
        record = Prediction(
            review_text=r["review_text"],
            sentiment_label=r["sentiment_label"],
            sentiment_confidence=r["sentiment_confidence"],
            category_label=r["category_label"],
            category_confidence=r["category_confidence"],
        )
        db.add(record)
        records.append(record)
    try:
        db.commit()
        for rec in records:
            db.refresh(rec)
        return records
    except SQLAlchemyError as exc:
        db.rollback()
        logger.warning("Could not save batch predictions: %s", exc)
        return []


def query_predictions(db: Session, skip: int = 0, limit: int = 100) -> tuple[int, list]:
    """Return total count and paginated predictions, tolerating DB errors."""
    try:
        total = db.query(Prediction).count()
        items = (
            db.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return total, items
    except SQLAlchemyError as exc:
        logger.warning("Could not query predictions: %s", exc)
        return 0, []
