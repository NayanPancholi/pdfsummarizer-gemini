"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import GeminiConfig  # noqa: E402


@pytest.fixture
def config() -> GeminiConfig:
    return GeminiConfig(
        api_key="test-key",
        model="test-model",
        temperature=0,
        max_output_tokens=100,
        chunk_size=200,
        chunk_overlap=20,
    )
