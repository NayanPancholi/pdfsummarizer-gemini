"""Extract text from PDF files using pypdf."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Mapping

from pypdf import PdfReader


@dataclass
class PdfDocument:
    """Text extracted from a PDF, one entry per page."""

    pages: list[tuple[int, str]] = field(default_factory=list)
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def non_empty_pages(self) -> int:
        return sum(1 for _, text in self.pages if text.strip())

    @property
    def text(self) -> str:
        """All pages joined with page markers for traceability."""
        return "\n\n".join(
            f"[Page {number}]\n{text}" for number, text in self.pages if text.strip()
        )


def load_pdf(source: str | Path | BinaryIO) -> PdfDocument:
    """Read a PDF and return its extracted text.

    Args:
        source: A file path or a binary file-like object (e.g. ``BytesIO``).

    Raises:
        ValueError: If the PDF has no extractable text.
    """
    reader = PdfReader(source)
    pages = [
        (number, (page.extract_text() or "").strip())
        for number, page in enumerate(reader.pages, start=1)
    ]

    document = PdfDocument(pages=pages, metadata=dict(reader.metadata or {}))
    if document.non_empty_pages == 0:
        raise ValueError(
            "No text could be extracted from this PDF. It may be scanned "
            "images or protected against copying."
        )
    return document
