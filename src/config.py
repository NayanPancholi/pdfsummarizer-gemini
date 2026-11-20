"""Central configuration loaded from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GeminiConfig:
    """Settings required to talk to Gemini and chunk a document."""

    api_key: str
    model: str
    temperature: float = 0.2
    max_output_tokens: int = 2048
    chunk_size: int = 4000
    chunk_overlap: int = 400


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def load_config(env_path: Path | None = None) -> GeminiConfig:
    """Load and validate configuration from ``.env``.

    Args:
        env_path: Optional explicit path to an env file. Defaults to the
            project-root ``.env``.

    Raises:
        ValueError: If required values are missing or malformed.
    """
    load_dotenv(env_path or PROJECT_ROOT / ".env", override=False)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()

    missing = []
    if not api_key:
        missing.append("GEMINI_API_KEY")
    if not model:
        missing.append("GEMINI_MODEL")
    if missing:
        raise ValueError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your values."
        )

    return GeminiConfig(
        api_key=api_key,
        model=model,
        chunk_size=_positive_int("PDF_CHUNK_SIZE", 4000),
        chunk_overlap=_positive_int("PDF_CHUNK_OVERLAP", 400),
    )
