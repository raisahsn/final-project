"""Tests for text preprocessing utilities."""

import pytest

from tokopedia_ml.preprocessing import clean_text, validate_input_text


def test_clean_text_lowercases_and_removes_urls():
    text = "Barangnya BAGUS banget! Cek http://toko.com ya"
    cleaned = clean_text(text)
    assert "bagus" in cleaned
    assert "http" not in cleaned
    assert "BAGUS" not in cleaned
    assert "!" not in cleaned


def test_clean_text_removes_stopwords():
    text = "saya sangat suka dengan produk ini dan itu"
    cleaned = clean_text(text)
    assert "saya" not in cleaned
    assert "sangat" not in cleaned
    assert "dan" not in cleaned
    assert "suka" in cleaned
    assert "produk" in cleaned


def test_validate_input_text_rejects_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_input_text("   ")


def test_validate_input_text_rejects_none():
    with pytest.raises(ValueError, match="cannot be None"):
        validate_input_text(None)


def test_validate_input_text_enforces_max_length():
    with pytest.raises(ValueError, match="exceeds maximum length"):
        validate_input_text("x" * 2001)
