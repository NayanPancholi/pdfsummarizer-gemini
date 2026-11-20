"""LangChain LCEL map-reduce summarization pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.chunker import count_tokens, split_text
from src.config import GeminiConfig, load_config

SUMMARY_STYLES: dict[str, str] = {
    "concise": "A short paragraph capturing the main point of the document.",
    "detailed": "A thorough, section-by-section summary retaining key facts and figures.",
    "bullets": "A bulleted list of the most important points and conclusions.",
    "insights": "Key findings, decisions, and actionable insights from the document.",
}

_MAP_TEMPLATE = """You are part of a pipeline that summarizes a PDF document.

Summarize this single part using the requested style:
{style_instruction}

Part {part_number} of {total_parts}.

Document part:
{text}

Rules:
- Use only information present in this part.
- Do not invent facts, names, numbers, or conclusions.
- If the part has no meaningful content, reply with: (no content)

Summary of this part:"""

_REDUCE_TEMPLATE = """You are the final step of a pipeline that summarizes a PDF document.

You will receive {part_count} partial summaries, each covering one part of the same document.
Combine them into one coherent summary using this style:
{style_instruction}

Partial summaries:
{summaries}

Rules:
- Remove repetition and merge related ideas.
- Preserve important facts, figures, and conclusions.
- Do not add information that is not present in the partial summaries.

Final summary:"""


@dataclass
class SummaryResult:
    """A generated summary plus useful statistics."""

    summary: str
    chunk_count: int
    input_tokens: int


class PdfSummarizer:
    """Split a document, summarize each part, then merge the results."""

    def __init__(self, config: GeminiConfig | None = None, llm: BaseChatModel | None = None):
        self.config = config or load_config()
        self.llm = llm or self._build_llm()

    def _build_llm(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.config.model,
            google_api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )

    def summarize(self, text: str, style: str = "concise") -> SummaryResult:
        """Summarize extracted PDF text in the requested style."""
        if style not in SUMMARY_STYLES:
            raise ValueError(
                f"Unknown summary style {style!r}. Choose from: {', '.join(SUMMARY_STYLES)}"
            )

        chunks = split_text(text, self.config)
        if not chunks:
            raise ValueError("There is no text to summarize.")

        style_instruction = SUMMARY_STYLES[style]
        map_chain = (
            ChatPromptTemplate.from_template(_MAP_TEMPLATE)
            | self.llm
            | StrOutputParser()
        )
        reduce_chain = (
            ChatPromptTemplate.from_template(_REDUCE_TEMPLATE)
            | self.llm
            | StrOutputParser()
        )

        partials = map_chain.batch(
            [
                {
                    "text": chunk,
                    "style_instruction": style_instruction,
                    "part_number": number,
                    "total_parts": len(chunks),
                }
                for number, chunk in enumerate(chunks, start=1)
            ],
            config={"max_concurrency": 4},
        )
        partials = [part.strip() for part in partials if part and part.strip()]
        if not partials:
            raise ValueError("The model did not produce any summary content.")

        if len(partials) == 1:
            summary = partials[0]
        else:
            summary = reduce_chain.invoke(
                {
                    "summaries": "\n\n".join(f"- {part}" for part in partials),
                    "part_count": len(partials),
                    "style_instruction": style_instruction,
                }
            )

        return SummaryResult(
            summary=summary.strip(),
            chunk_count=len(chunks),
            input_tokens=count_tokens(text),
        )
