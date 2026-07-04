from __future__ import annotations

import re
from dataclasses import dataclass, replace

from nextsearch.ingestion.models import DocumentSection, MarkdownDocument


DEFAULT_MAX_SECTION_CHARS = 12_000

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PAGE_ANCHOR_RE = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->$")


@dataclass
class _SectionDraft:
    heading: str
    heading_path: tuple[str, ...]
    lines: list[tuple[str, int | None]]


def split_markdown_into_sections(
    document: MarkdownDocument,
    *,
    max_section_chars: int = DEFAULT_MAX_SECTION_CHARS,
) -> list[DocumentSection]:
    if max_section_chars < 1:
        raise ValueError("max_section_chars must be at least 1")

    drafts = _section_drafts(document.markdown)
    sections: list[DocumentSection] = []

    for index, draft in enumerate(drafts, start=1):
        section = _draft_to_section(f"section-{index:04d}", draft)
        sections.extend(_split_large_section(section, max_section_chars))

    return sections


def _section_drafts(markdown: str) -> list[_SectionDraft]:
    drafts: list[_SectionDraft] = []
    heading_stack: list[str] = []
    draft = _SectionDraft(heading="Untitled", heading_path=("Untitled",), lines=[])
    pending_page_anchors: list[tuple[str, int | None]] = []
    current_page: int | None = None

    for line in markdown.splitlines():
        page_match = _PAGE_ANCHOR_RE.match(line)
        if page_match is not None:
            current_page = int(page_match.group(1))
            pending_page_anchors.append((line, current_page))
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match is not None:
            if _has_meaningful_content(draft.lines):
                drafts.append(draft)
                prefix_lines = pending_page_anchors
            else:
                prefix_lines = draft.lines + pending_page_anchors

            pending_page_anchors = []
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading)
            draft = _SectionDraft(
                heading=heading,
                heading_path=tuple(heading_stack),
                lines=prefix_lines + [(line, current_page)],
            )
            continue

        if pending_page_anchors:
            draft.lines.extend(pending_page_anchors)
            pending_page_anchors = []
        draft.lines.append((line, current_page))

    if pending_page_anchors:
        draft.lines.extend(pending_page_anchors)

    if _has_meaningful_content(draft.lines):
        drafts.append(draft)

    return drafts


def _draft_to_section(section_id: str, draft: _SectionDraft) -> DocumentSection:
    page_numbers = [
        page_number for _, page_number in draft.lines if page_number is not None
    ]
    text = "\n".join(line for line, _ in draft.lines).strip()
    return DocumentSection(
        id=section_id,
        heading=draft.heading,
        heading_path=draft.heading_path,
        page_start=min(page_numbers) if page_numbers else None,
        page_end=max(page_numbers) if page_numbers else None,
        text=text,
    )


def _split_large_section(
    section: DocumentSection,
    max_section_chars: int,
) -> list[DocumentSection]:
    if len(section.text) <= max_section_chars:
        return [section]

    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0

    for paragraph in _paragraphs(section.text):
        paragraph_chars = len(paragraph)
        if paragraph_chars > max_section_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_chars = 0
            chunks.extend(_hard_split(paragraph, max_section_chars))
            continue

        separator_chars = 2 if current else 0
        if current and current_chars + separator_chars + paragraph_chars > max_section_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_chars = 0

        current.append(paragraph)
        current_chars += separator_chars + paragraph_chars

    if current:
        chunks.append("\n\n".join(current))

    if len(chunks) <= 1:
        return [replace(section, text=chunks[0] if chunks else section.text)]

    return [
        replace(section, id=f"{section.id}-part-{index:04d}", text=chunk)
        for index, chunk in enumerate(chunks, start=1)
    ]


def _paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


def _hard_split(text: str, max_chars: int) -> list[str]:
    return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]


def _has_meaningful_content(lines: list[tuple[str, int | None]]) -> bool:
    return any(
        line.strip() and _PAGE_ANCHOR_RE.match(line) is None for line, _ in lines
    )
