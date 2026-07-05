"""Tokopedia ML package for review sentiment and category classification."""

from .inference import predict_text, predict_batch

__all__ = ["predict_text", "predict_batch"]
