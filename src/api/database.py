"""Database layer for persisting predictions."""

from datetime import datetime
from typing import Generator

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from tokopedia_ml import config

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
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session for FastAPI dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(db: Session, result: dict) -> Prediction:
    """Persist a single prediction result."""
    record = Prediction(
        review_text=result["review_text"],
        sentiment_label=result["sentiment"]["label"],
        sentiment_confidence=result["sentiment"]["confidence"],
        category_label=result["category"]["label"],
        category_confidence=result["category"]["confidence"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def save_batch_predictions(db: Session, results: list) -> list:
    """Persist a batch of prediction results."""
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
    db.commit()
    for rec in records:
        db.refresh(rec)
    return records
