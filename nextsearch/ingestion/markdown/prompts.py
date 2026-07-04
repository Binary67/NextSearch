from __future__ import annotations

from collections.abc import Sequence

from nextsearch.ingestion.models import PageContent
from nextsearch.llm.types import LLMMessage


SYSTEM_PROMPT = """You convert PDF page text into faithful Markdown.
Preserve the document's headings, subheadings, lists, tables, captions, and values.
Do not summarize, omit facts, invent content, or add commentary.
Keep each page anchor exactly as provided before that page's content."""


def build_markdown_extraction_messages(
    pages: Sequence[PageContent],
) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(role="user", content=_build_user_prompt(pages)),
    ]


def _build_user_prompt(pages: Sequence[PageContent]) -> str:
    page_blocks = "\n\n".join(
        f"<!-- page: {page.page_number} -->\n{page.text}" for page in pages
    )
    return (
        "Convert these PDF pages to Markdown. Preserve each page anchor exactly.\n\n"
        f"{page_blocks}"
    )
