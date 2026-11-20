"""Token-aware text splitting for long PDFs."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import GeminiConfig

_ENCODING = "cl100k_base"


def count_tokens(text: str) -> int:
    """Approximate the token count used for chunking and statistics."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding(_ENCODING).encode(text))
    except Exception:
        # The encoding file is downloaded on first use; fall back to a
        # rough estimate when offline so the app still works.
        return max(1, len(text) // 4)


def _build_splitter(config: GeminiConfig) -> RecursiveCharacterTextSplitter:
    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=_ENCODING,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    except Exception:
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )


def split_text(text: str, config: GeminiConfig) -> list[str]:
    """Split text into overlapping chunks that fit the model context."""
    if not text or not text.strip():
        return []
    return [chunk for chunk in _build_splitter(config).split_text(text) if chunk.strip()]
