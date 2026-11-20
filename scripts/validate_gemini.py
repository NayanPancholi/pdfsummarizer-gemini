"""Validate the Gemini API key with a real, minimal model call."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import GeminiConfig, load_config  # noqa: E402
from src.summarizer import PdfSummarizer  # noqa: E402

_PROBE_PROMPT = "Reply with exactly the single word: OK"


def _mask(secret: str) -> str:
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}...{secret[-4:]}"


def _redact(message: str, secret: str) -> str:
    return message.replace(secret, "***") if secret else message


def validate(config: GeminiConfig | None = None) -> bool:
    """Return True when the configured API key and model work."""
    print("[Gemini] Checking configuration...")
    try:
        cfg = config or load_config()
    except Exception as exc:
        print(f"[Gemini] FAIL: {exc}")
        return False

    print(f"[Gemini] API key: {_mask(cfg.api_key)} | model: {cfg.model}")
    try:
        llm = PdfSummarizer(config=cfg).llm
        response = llm.invoke(_PROBE_PROMPT)
        content = response.content
        if isinstance(content, list):
            content = " ".join(str(getattr(part, "text", part)) for part in content)
        reply = str(content).strip().lower()
        if "ok" not in reply:
            print(f"[Gemini] FAIL: unexpected reply {reply[:120]!r}")
            return False
    except Exception as exc:
        detail = _redact(f"{type(exc).__name__}: {exc}", cfg.api_key)
        print(f"[Gemini] FAIL: API call error - {detail}")
        return False

    print("[Gemini] PASS: API key is valid and the model responded.")
    return True


if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
