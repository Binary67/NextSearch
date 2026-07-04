from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageContent:
    page_number: int
    text: str


@dataclass(frozen=True)
class MarkdownBatch:
    page_start: int
    page_end: int
    markdown: str
    usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class MarkdownDocument:
    markdown: str
    source_path: Path
    page_count: int
    batches: tuple[MarkdownBatch, ...]
