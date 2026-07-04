from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from nextsearch.ingestion.errors import MarkdownExtractionError
from nextsearch.ingestion.markdown.prompts import build_markdown_extraction_messages
from nextsearch.ingestion.markdown.stitcher import stitch_markdown_batches
from nextsearch.ingestion.models import MarkdownBatch, MarkdownDocument, PageContent
from nextsearch.llm.service import LLMService


DEFAULT_MAX_PAGES_PER_BATCH = 5
DEFAULT_MAX_BATCH_CHARS = 12_000


def extract_markdown(
    *,
    pages: Sequence[PageContent],
    llm: LLMService,
    source_path: Path,
    role: str = "markdown_extraction",
    max_pages_per_batch: int = DEFAULT_MAX_PAGES_PER_BATCH,
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
) -> MarkdownDocument:
    page_batches = split_pages_for_markdown(
        pages,
        max_pages_per_batch=max_pages_per_batch,
        max_batch_chars=max_batch_chars,
    )
    markdown_batches: list[MarkdownBatch] = []

    for page_batch in page_batches:
        page_start = page_batch[0].page_number
        page_end = page_batch[-1].page_number
        messages = build_markdown_extraction_messages(page_batch)

        try:
            response = llm.generate_text(
                role=role,
                messages=messages,
                temperature=0,
            )
        except Exception as exc:
            raise MarkdownExtractionError(
                f"Markdown extraction failed for pages {page_start}-{page_end}"
            ) from exc

        _require_page_anchors(response.text, page_batch, page_start, page_end)
        markdown_batches.append(
            MarkdownBatch(
                page_start=page_start,
                page_end=page_end,
                markdown=response.text,
                usage=response.usage,
            )
        )

    return MarkdownDocument(
        markdown=stitch_markdown_batches(markdown_batches),
        source_path=Path(source_path),
        page_count=len({page.page_number for page in pages}),
        batches=tuple(markdown_batches),
    )


def split_pages_for_markdown(
    pages: Sequence[PageContent],
    *,
    max_pages_per_batch: int = DEFAULT_MAX_PAGES_PER_BATCH,
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
) -> list[tuple[PageContent, ...]]:
    if max_pages_per_batch < 1:
        raise ValueError("max_pages_per_batch must be at least 1")
    if max_batch_chars < 1:
        raise ValueError("max_batch_chars must be at least 1")

    batches: list[tuple[PageContent, ...]] = []
    current: list[PageContent] = []
    current_chars = 0

    for page in pages:
        page_parts = _split_large_page(page, max_batch_chars)
        for part in page_parts:
            part_chars = len(part.text)
            batch_full = (
                len(current) >= max_pages_per_batch
                or current_chars + part_chars > max_batch_chars
            )
            if current and batch_full:
                batches.append(tuple(current))
                current = []
                current_chars = 0

            current.append(part)
            current_chars += part_chars

    if current:
        batches.append(tuple(current))

    return batches


def _split_large_page(page: PageContent, max_chars: int) -> list[PageContent]:
    if len(page.text) <= max_chars:
        return [page]

    parts: list[PageContent] = []
    current: list[str] = []
    current_chars = 0

    for paragraph in _paragraphs(page.text):
        paragraph_chars = len(paragraph)
        if paragraph_chars > max_chars:
            if current:
                parts.append(
                    PageContent(
                        page_number=page.page_number,
                        text="\n\n".join(current),
                    )
                )
                current = []
                current_chars = 0
            parts.extend(
                PageContent(page_number=page.page_number, text=chunk)
                for chunk in _hard_split(paragraph, max_chars)
            )
            continue

        separator_chars = 2 if current else 0
        if current and current_chars + separator_chars + paragraph_chars > max_chars:
            parts.append(
                PageContent(
                    page_number=page.page_number,
                    text="\n\n".join(current),
                )
            )
            current = []
            current_chars = 0

        current.append(paragraph)
        current_chars += separator_chars + paragraph_chars

    if current:
        parts.append(
            PageContent(
                page_number=page.page_number,
                text="\n\n".join(current),
            )
        )

    return parts


def _paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


def _hard_split(text: str, max_chars: int) -> list[str]:
    return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]


def _require_page_anchors(
    markdown: str,
    pages: Sequence[PageContent],
    page_start: int,
    page_end: int,
) -> None:
    missing = [
        page_number
        for page_number in sorted({page.page_number for page in pages})
        if f"<!-- page: {page_number} -->" not in markdown
    ]
    if missing:
        page_list = ", ".join(str(page_number) for page_number in missing)
        raise MarkdownExtractionError(
            "Markdown extraction for pages "
            f"{page_start}-{page_end} omitted page anchors: {page_list}"
        )
