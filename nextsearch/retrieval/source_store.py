from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Protocol

from nextsearch.ingestion.markdown.sections import split_markdown_into_sections
from nextsearch.ingestion.models import DocumentSection, MarkdownDocument


class SourceStore(Protocol):
    def get_section(
        self,
        *,
        document_id: str,
        section_id: str,
    ) -> DocumentSection | None:
        ...


@dataclass(frozen=True)
class MarkdownArtifactSourceStore:
    artifacts_root: Path

    def get_section(
        self,
        *,
        document_id: str,
        section_id: str,
    ) -> DocumentSection | None:
        return self._sections_by_id(document_id).get(section_id)

    @cache
    def _sections_by_id(self, document_id: str) -> dict[str, DocumentSection]:
        document_dir = Path(self.artifacts_root) / "documents" / document_id
        markdown_path = document_dir / "document.md"
        manifest_path = document_dir / "manifest.json"
        if not markdown_path.exists() or not manifest_path.exists():
            return {}

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        document = MarkdownDocument(
            markdown=markdown_path.read_text(encoding="utf-8"),
            source_path=Path(manifest["source_path"]),
            page_count=manifest["page_count"],
            batches=(),
        )
        return {
            section.id: section
            for section in split_markdown_into_sections(document)
        }
