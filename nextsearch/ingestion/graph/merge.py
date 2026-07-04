from __future__ import annotations

from nextsearch.ingestion.graph.models import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    SourceRef,
)


CORPUS_DOCUMENT_ID = "corpus"


def empty_corpus_graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        document_id=CORPUS_DOCUMENT_ID,
        content_hash="",
        source_path=CORPUS_DOCUMENT_ID,
        page_count=0,
        nodes=[],
        edges=[],
    )


def merge_knowledge_graphs(
    existing: KnowledgeGraph,
    incoming: KnowledgeGraph,
) -> KnowledgeGraph:
    nodes = {node.id: node for node in existing.nodes}
    edges = {edge.id: edge for edge in existing.edges}

    for incoming_node in incoming.nodes:
        existing_node = nodes.get(incoming_node.id)
        if existing_node is None:
            nodes[incoming_node.id] = incoming_node
            continue

        nodes[incoming_node.id] = existing_node.model_copy(
            update={
                "aliases": _merge_aliases(existing_node.aliases, incoming_node.aliases),
                "description": existing_node.description or incoming_node.description,
                "source_refs": merge_source_refs(
                    existing_node.source_refs,
                    incoming_node.source_refs,
                ),
            }
        )

    for incoming_edge in incoming.edges:
        existing_edge = edges.get(incoming_edge.id)
        if existing_edge is None:
            edges[incoming_edge.id] = incoming_edge
            continue

        edges[incoming_edge.id] = existing_edge.model_copy(
            update={
                "description": existing_edge.description or incoming_edge.description,
                "confidence": max(existing_edge.confidence, incoming_edge.confidence),
                "source_refs": merge_source_refs(
                    existing_edge.source_refs,
                    incoming_edge.source_refs,
                ),
            }
        )

    return existing.model_copy(
        update={
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
        }
    )


def remove_document_from_graph(
    graph: KnowledgeGraph,
    document_id: str,
) -> KnowledgeGraph:
    nodes: list[GraphNode] = []
    for node in graph.nodes:
        source_refs = _remove_document_refs(node.source_refs, document_id)
        if source_refs:
            nodes.append(node.model_copy(update={"source_refs": source_refs}))

    node_ids = {node.id for node in nodes}
    edges: list[GraphEdge] = []
    for edge in graph.edges:
        source_refs = _remove_document_refs(edge.source_refs, document_id)
        if (
            source_refs
            and edge.source_node_id in node_ids
            and edge.target_node_id in node_ids
        ):
            edges.append(edge.model_copy(update={"source_refs": source_refs}))

    return graph.model_copy(update={"nodes": nodes, "edges": edges})


def merge_source_refs(
    existing: list[SourceRef],
    incoming: list[SourceRef],
) -> list[SourceRef]:
    merged = list(existing)
    seen = {_source_ref_key(source_ref) for source_ref in existing}
    for source_ref in incoming:
        key = _source_ref_key(source_ref)
        if key not in seen:
            merged.append(source_ref)
            seen.add(key)
    return merged


def _merge_aliases(existing: list[str], incoming: list[str]) -> list[str]:
    aliases = list(existing)
    seen = {alias.strip().lower() for alias in existing}
    for alias in incoming:
        cleaned = alias.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            aliases.append(cleaned)
            seen.add(key)
    return aliases


def _remove_document_refs(
    source_refs: list[SourceRef],
    document_id: str,
) -> list[SourceRef]:
    return [
        source_ref
        for source_ref in source_refs
        if source_ref.document_id != document_id
    ]


def _source_ref_key(source_ref: SourceRef) -> tuple[str, str, str, int | None, int | None, str]:
    return (
        source_ref.document_id,
        source_ref.section_id,
        source_ref.heading,
        source_ref.page_start,
        source_ref.page_end,
        source_ref.quote,
    )
