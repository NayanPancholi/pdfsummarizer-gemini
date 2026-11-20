"""Tests for the LangChain map-reduce pipeline."""

from __future__ import annotations

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.config import GeminiConfig
from src.summarizer import PdfSummarizer


class _FakeChatModel(BaseChatModel):
    """Deterministic stand-in for Gemini that needs no network access."""

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt = messages[0].content
        content = "FINAL SUMMARY" if "partial summaries" in prompt else "PART SUMMARY"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def test_summarize_uses_map_reduce_for_long_text(config: GeminiConfig):
    result = PdfSummarizer(config=config, llm=_FakeChatModel()).summarize("word " * 2000)

    assert result.chunk_count > 1
    assert result.summary == "FINAL SUMMARY"
    assert result.input_tokens > 0


def test_summarize_returns_part_summary_for_single_chunk(config: GeminiConfig):
    result = PdfSummarizer(config=config, llm=_FakeChatModel()).summarize("hello world")

    assert result.chunk_count == 1
    assert result.summary == "PART SUMMARY"


def test_summarize_rejects_unknown_style(config: GeminiConfig):
    summarizer = PdfSummarizer(config=config, llm=_FakeChatModel())

    with pytest.raises(ValueError, match="Unknown summary style"):
        summarizer.summarize("hello world", style="unknown")
