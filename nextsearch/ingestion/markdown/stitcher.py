from __future__ import annotations

from collections.abc import Sequence

from nextsearch.ingestion.models import MarkdownBatch


def stitch_markdown_batches(batches: Sequence[MarkdownBatch]) -> str:
    markdown = "\n\n".join(
        batch.markdown.strip() for batch in batches if batch.markdown.strip()
    )
    if markdown:
        return markdown + "\n"
    return ""
