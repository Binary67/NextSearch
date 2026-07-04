from __future__ import annotations

from pathlib import Path

from nextsearch.ingestion.artifacts import write_markdown_artifacts
from nextsearch.ingestion.markdown.llm_extractor import extract_markdown
from nextsearch.ingestion.models import MarkdownDocument
from nextsearch.ingestion.sources.pdf import extract_pdf_pages
from nextsearch.llm.service import LLMService


def extract_pdf_to_markdown(
    pdf_path: Path,
    llm: LLMService,
    *,
    output_dir: Path | None = None,
) -> MarkdownDocument:
    source_path = Path(pdf_path)
    pages = extract_pdf_pages(source_path)
    document = extract_markdown(
        pages=pages,
        llm=llm,
        source_path=source_path,
    )

    if output_dir is not None:
        write_markdown_artifacts(document, Path(output_dir))

    return document
