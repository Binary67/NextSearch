from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from nextsearch.ingestion.errors import PDFExtractionError
from nextsearch.ingestion.models import PageContent


def extract_pdf_pages(pdf_path: Path) -> list[PageContent]:
    path = Path(pdf_path)
    if not path.exists():
        raise PDFExtractionError(f"PDF file not found: {path}")
    if not path.is_file():
        raise PDFExtractionError(f"PDF path is not a file: {path}")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise PDFExtractionError(f"Invalid PDF file: {path}") from exc

    if len(reader.pages) == 0:
        raise PDFExtractionError(f"PDF contains no pages: {path}")

    pages: list[PageContent] = []
    unreadable_pages: list[int] = []

    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise PDFExtractionError(
                f"Failed to extract text from PDF page {index}: {path}"
            ) from exc

        normalized_text = _normalize_text(text)
        if not normalized_text.strip():
            unreadable_pages.append(index)

        pages.append(PageContent(page_number=index, text=normalized_text))

    if unreadable_pages:
        page_list = ", ".join(str(page) for page in unreadable_pages)
        raise PDFExtractionError(
            "PDF contains pages with no extractable text; scanned/image PDFs "
            f"are unsupported in v1: pages {page_list}"
        )

    return pages


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()
