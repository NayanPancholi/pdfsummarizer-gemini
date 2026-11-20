"""Tests for PDF text extraction."""

from __future__ import annotations

import pytest

from src.pdf_loader import load_pdf


class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, source):
        self.pages = [_FakePage("First page text"), _FakePage("   ")]
        self.metadata = {"/Title": "Test document"}


def test_load_pdf_extracts_pages(monkeypatch):
    monkeypatch.setattr("src.pdf_loader.PdfReader", _FakeReader)
    document = load_pdf("ignored.pdf")

    assert document.page_count == 2
    assert document.non_empty_pages == 1
    assert "[Page 1]" in document.text
    assert "First page text" in document.text
    assert document.metadata["/Title"] == "Test document"


def test_load_pdf_rejects_document_without_text(monkeypatch):
    class EmptyReader:
        def __init__(self, source):
            self.pages = [_FakePage("")]
            self.metadata = None

    monkeypatch.setattr("src.pdf_loader.PdfReader", EmptyReader)
    with pytest.raises(ValueError, match="No text could be extracted"):
        load_pdf("empty.pdf")
