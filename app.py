"""Streamlit web UI for the LangChain PDF summarizer."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config  # noqa: E402
from src.pdf_loader import load_pdf  # noqa: E402
from src.summarizer import SUMMARY_STYLES, PdfSummarizer  # noqa: E402

st.set_page_config(page_title="PDF Summarizer", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_summarizer() -> PdfSummarizer:
    """Create one cached summarizer for the whole Streamlit session."""
    return PdfSummarizer(config=load_config())


def main() -> None:
    st.title("📄 PDF Summarizer")
    st.caption("Upload a PDF and generate a summary with LangChain + Gemini.")

    with st.sidebar:
        st.header("Settings")
        style = st.selectbox("Summary style", list(SUMMARY_STYLES), index=0)
        st.caption(SUMMARY_STYLES[style])
        st.divider()
        st.markdown("Set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env` before running.")

    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded is None:
        st.info("Upload a PDF to get started.")
        return

    if st.button("Summarize", type="primary"):
        try:
            with st.spinner("Reading PDF..."):
                document = load_pdf(io.BytesIO(uploaded.getvalue()))
            st.caption(
                f"{document.page_count} page(s) · "
                f"{document.non_empty_pages} page(s) with extractable text"
            )
            with st.spinner("Generating summary..."):
                result = get_summarizer().summarize(document.text, style=style)
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Summarization failed: {type(exc).__name__}: {exc}")
            return

        st.subheader("Summary")
        st.markdown(result.summary)
        st.caption(
            f"{result.chunk_count} chunk(s) · ~{result.input_tokens:,} input tokens"
        )
        st.download_button(
            "Download summary (.md)",
            data=result.summary,
            file_name=f"{Path(uploaded.name).stem}_summary.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
