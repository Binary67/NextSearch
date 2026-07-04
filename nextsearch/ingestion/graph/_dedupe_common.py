from __future__ import annotations

import re
from typing import NamedTuple

from nextsearch.ingestion.graph.models import GraphNode, SourceRef


MERGE_CONFIDENCE_THRESHOLD = 0.9
TOKEN_SIMILARITY_THRESHOLD = 0.88
SEMANTIC_SIMILARITY_THRESHOLD = 0.88
MAX_SEMANTIC_MATCHES_PER_NODE = 5
MAX_SOURCE_QUOTES = 4
MAX_QUOTE_CHARS = 260
MAX_EDGE_SUMMARIES = 8

ORGANIZATION_SUFFIX_TOKENS = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "ltd",
    "plc",
    "pte",
}


class _MergeCandidate(NamedTuple):
    source_node_id: str
    target_node_id: str
    reasons: tuple[str, ...]


class _UnionFind:
    def __init__(self, node_ids: list[str]) -> None:
        self._parent = {node_id: node_id for node_id in node_ids}

    def find(self, node_id: str) -> str:
        parent = self._parent[node_id]
        if parent != node_id:
            self._parent[node_id] = self.find(parent)
        return self._parent[node_id]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


def _candidate_key(left_id: str, right_id: str) -> tuple[str, str]:
    return (left_id, right_id) if left_id < right_id else (right_id, left_id)


def _merge_source_refs(
    existing: list[SourceRef],
    incoming: list[SourceRef],
) -> list[SourceRef]:
    merged = list(existing)
    seen = {
        (
            source_ref.document_id,
            source_ref.section_id,
            source_ref.heading,
            source_ref.page_start,
            source_ref.page_end,
            source_ref.quote,
        )
        for source_ref in existing
    }
    for source_ref in incoming:
        key = (
            source_ref.document_id,
            source_ref.section_id,
            source_ref.heading,
            source_ref.page_start,
            source_ref.page_end,
            source_ref.quote,
        )
        if key not in seen:
            merged.append(source_ref)
            seen.add(key)
    return merged


def _node_names(node: GraphNode) -> list[str]:
    return [node.name, *node.aliases]


def _normalize_text(value: str) -> str:
    return " ".join(_name_tokens(value))


def _name_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _source_quote_lines(source_refs: list[SourceRef]) -> list[str]:
    lines: list[str] = []
    for source_ref in source_refs[:MAX_SOURCE_QUOTES]:
        quote = source_ref.quote.strip()
        if len(quote) > MAX_QUOTE_CHARS:
            quote = quote[:MAX_QUOTE_CHARS].rstrip() + "..."
        lines.append(f"- {quote}")
    return lines
