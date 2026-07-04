from __future__ import annotations

from nextsearch.agent.models import Citation, EvidenceRecord
from nextsearch.ingestion.graph.models import SourceRef
from nextsearch.retrieval.graph_search import GraphSearchResult
from nextsearch.retrieval.source_store import SourceStore


def build_evidence_records(
    result: GraphSearchResult,
    source_store: SourceStore,
    *,
    start_id: int = 1,
) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    record_by_key: dict[tuple[str, str, str, str], EvidenceRecord] = {}

    for node in result.nodes:
        for source_ref in node.source_refs:
            _add_record(
                records=records,
                record_by_key=record_by_key,
                source_store=source_store,
                source_ref=source_ref,
                start_id=start_id,
                graph_node_id=node.id,
                graph_edge_id=None,
            )

    for edge in result.edges:
        for source_ref in edge.source_refs:
            _add_record(
                records=records,
                record_by_key=record_by_key,
                source_store=source_store,
                source_ref=source_ref,
                start_id=start_id,
                graph_node_id=None,
                graph_edge_id=edge.id,
            )

    return records


def _add_record(
    *,
    records: list[EvidenceRecord],
    record_by_key: dict[tuple[str, str, str, str], EvidenceRecord],
    source_store: SourceStore,
    source_ref: SourceRef,
    start_id: int,
    graph_node_id: str | None,
    graph_edge_id: str | None,
) -> None:
    section = source_store.get_section(
        document_id=source_ref.document_id,
        section_id=source_ref.section_id,
    )
    if section is None:
        return

    snippet = _snippet(section.text, source_ref.quote)
    key = (
        source_ref.document_id,
        source_ref.section_id,
        source_ref.quote,
        snippet,
    )
    record = record_by_key.get(key)
    if record is None:
        record = EvidenceRecord(
            evidence_id=start_id + len(records),
            citation=Citation(
                document_id=source_ref.document_id,
                section_id=source_ref.section_id,
                heading=source_ref.heading,
                page_start=source_ref.page_start,
                page_end=source_ref.page_end,
                quote=source_ref.quote,
            ),
            snippet=snippet,
            graph_node_ids=[],
            graph_edge_ids=[],
        )
        records.append(record)
        record_by_key[key] = record

    if graph_node_id is not None and graph_node_id not in record.graph_node_ids:
        record.graph_node_ids.append(graph_node_id)
    if graph_edge_id is not None and graph_edge_id not in record.graph_edge_ids:
        record.graph_edge_ids.append(graph_edge_id)


def _snippet(text: str, quote: str, *, context_chars: int = 350) -> str:
    clean_text = text.strip()
    clean_quote = quote.strip()
    if clean_quote == "":
        return clean_text[: context_chars * 2].strip()

    index = clean_text.find(clean_quote)
    if index < 0:
        return clean_text[: context_chars * 2].strip()

    start = max(0, index - context_chars)
    end = min(len(clean_text), index + len(clean_quote) + context_chars)
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(clean_text) else ""
    return f"{prefix}{clean_text[start:end].strip()}{suffix}"
