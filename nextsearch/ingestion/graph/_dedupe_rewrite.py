from __future__ import annotations

from collections import defaultdict

from nextsearch.ingestion.graph._dedupe_common import (
    MERGE_CONFIDENCE_THRESHOLD,
    ORGANIZATION_SUFFIX_TOKENS,
    _UnionFind,
    _merge_source_refs,
    _name_tokens,
    _node_names,
    _normalize_text,
)
from nextsearch.ingestion.graph.llm_extractor import normalize_edge_id
from nextsearch.ingestion.graph.models import (
    GraphEdge,
    GraphNode,
    GraphNodeMergeDecision,
    KnowledgeGraph,
    SourceRef,
)


def _build_node_replacements(
    nodes: list[GraphNode],
    union_find: _UnionFind,
) -> tuple[dict[str, str], dict[str, str]]:
    grouped_nodes: dict[str, list[GraphNode]] = defaultdict(list)
    for node in nodes:
        grouped_nodes[union_find.find(node.id)].append(node)

    replacements: dict[str, str] = {}
    canonical_by_root: dict[str, str] = {}
    for root, group in grouped_nodes.items():
        canonical = _choose_canonical_node(group)
        canonical_by_root[root] = canonical.id
        for node in group:
            if node.id != canonical.id:
                replacements[node.id] = canonical.id

    return replacements, canonical_by_root


def _apply_node_replacements(
    *,
    graph: KnowledgeGraph,
    replacements: dict[str, str],
    canonical_by_root: dict[str, str],
    union_find: _UnionFind,
) -> KnowledgeGraph:
    merged_nodes_by_id = _merged_nodes_by_id(
        graph.nodes,
        canonical_by_root,
        union_find,
    )

    nodes: list[GraphNode] = []
    seen_node_ids: set[str] = set()
    for node in graph.nodes:
        canonical_id = replacements.get(node.id, node.id)
        if canonical_id in seen_node_ids:
            continue
        nodes.append(merged_nodes_by_id[canonical_id])
        seen_node_ids.add(canonical_id)

    edges_by_id: dict[str, GraphEdge] = {}
    for edge in graph.edges:
        source_node_id = replacements.get(edge.source_node_id, edge.source_node_id)
        target_node_id = replacements.get(edge.target_node_id, edge.target_node_id)
        if source_node_id == target_node_id and edge.source_node_id != edge.target_node_id:
            continue

        edge_id = normalize_edge_id(
            source_node_id,
            edge.relation_type,
            target_node_id,
        )
        updated_edge = edge.model_copy(
            update={
                "id": edge_id,
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
            }
        )
        existing = edges_by_id.get(edge_id)
        if existing is None:
            edges_by_id[edge_id] = updated_edge
            continue

        edges_by_id[edge_id] = existing.model_copy(
            update={
                "description": existing.description or updated_edge.description,
                "confidence": max(existing.confidence, updated_edge.confidence),
                "source_refs": _merge_source_refs(
                    existing.source_refs,
                    updated_edge.source_refs,
                ),
            }
        )

    return graph.model_copy(
        update={
            "nodes": nodes,
            "edges": list(edges_by_id.values()),
        }
    )


def _merged_nodes_by_id(
    nodes: list[GraphNode],
    canonical_by_root: dict[str, str],
    union_find: _UnionFind,
) -> dict[str, GraphNode]:
    grouped_nodes: dict[str, list[GraphNode]] = defaultdict(list)
    for node in nodes:
        canonical_id = canonical_by_root[union_find.find(node.id)]
        grouped_nodes[canonical_id].append(node)

    return {
        canonical_id: _merge_nodes(group)
        for canonical_id, group in grouped_nodes.items()
    }


def _merge_nodes(nodes: list[GraphNode]) -> GraphNode:
    canonical = _choose_canonical_node(nodes)
    return canonical.model_copy(
        update={
            "aliases": _merged_aliases(canonical, nodes),
            "description": canonical.description or _first_description(nodes),
            "source_refs": _merged_node_source_refs(nodes),
        }
    )


def _choose_canonical_node(nodes: list[GraphNode]) -> GraphNode:
    if len(nodes) == 1:
        return nodes[0]

    return min(nodes, key=_canonical_sort_key)


def _canonical_sort_key(node: GraphNode) -> tuple[int, int, int, str]:
    formal_name = _has_organization_suffix(node.name) if node.type == "organization" else False
    evidence_count = len(node.source_refs)
    acronym = _is_acronym_only(node.name)
    return (
        0 if formal_name else 1,
        -evidence_count,
        1 if acronym else -len(_normalize_text(node.name)),
        _normalize_text(node.name),
    )


def _merged_aliases(canonical: GraphNode, nodes: list[GraphNode]) -> list[str]:
    aliases: list[str] = []
    seen = {_normalize_text(canonical.name)}
    for value in [*canonical.aliases, *[name for node in nodes for name in _node_names(node)]]:
        cleaned = value.strip()
        key = _normalize_text(cleaned)
        if not cleaned or key in seen:
            continue
        aliases.append(cleaned)
        seen.add(key)
    return aliases


def _first_description(nodes: list[GraphNode]) -> str | None:
    for node in nodes:
        if node.description:
            return node.description
    return None


def _merged_node_source_refs(nodes: list[GraphNode]) -> list[SourceRef]:
    refs: list[SourceRef] = []
    for node in nodes:
        refs = _merge_source_refs(refs, node.source_refs)
    return refs


def _attach_canonical_node_id(
    decision: GraphNodeMergeDecision,
    union_find: _UnionFind,
    canonical_by_root: dict[str, str],
) -> GraphNodeMergeDecision:
    if (
        decision.decision != "same"
        or decision.confidence < MERGE_CONFIDENCE_THRESHOLD
    ):
        return decision

    return decision.model_copy(
        update={
            "canonical_node_id": canonical_by_root[
                union_find.find(decision.node_ids[0])
            ],
        }
    )


def _has_organization_suffix(value: str) -> bool:
    return any(token in ORGANIZATION_SUFFIX_TOKENS for token in _name_tokens(value))


def _is_acronym_only(value: str) -> bool:
    tokens = _name_tokens(value)
    return len(tokens) == 1 and 2 <= len(tokens[0]) <= 8
