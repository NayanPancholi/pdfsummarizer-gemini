"""Tests for token-aware splitting."""

from __future__ import annotations

from src.chunker import count_tokens, split_text
from src.config import GeminiConfig


def test_short_text_is_one_chunk(config: GeminiConfig):
    assert split_text("hello world", config) == ["hello world"]


def test_long_text_is_split(config: GeminiConfig):
    chunks = split_text("word " * 2000, config)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_empty_text_produces_no_chunks(config: GeminiConfig):
    assert split_text("   ", config) == []


def test_count_tokens_is_positive():
    assert count_tokens("hello world") > 0
