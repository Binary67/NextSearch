from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Sequence

from nextsearch.ingestion.graph.models import GraphEdge, GraphNode, KnowledgeGraph


@dataclass(frozen=True)
class GraphSearchResult:
    search_terms: tuple[str, ...]
    relation_types: tuple[str, ...]
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


def search_graph(
    graph: KnowledgeGraph,
    *,
    search_terms: Sequence[str],
    relation_types: Sequence[str] = (),
    max_nodes: int = 8,
    max_edges: int = 16,
) -> GraphSearchResult:
    terms = tuple(_dedupe(term.strip() for term in search_terms if term.strip()))
    normalized_terms = tuple(_normalize(term) for term in terms)
    relation_filter = set(relation_types)

    node_scores = [
        (_node_score(node, normalized_terms), node)
        for node in graph.nodes
    ]
    matched_nodes = [
        node
        for score, node in sorted(
            node_scores,
            key=lambda item: (-item[0], item[1].type, item[1].name.lower()),
        )
        if score > 0
    ][:max_nodes]
    matched_node_ids = {node.id for node in matched_nodes}

    candidate_edges = [
        edge
        for edge in graph.edges
        if _edge_matches(edge, normalized_terms, relation_filter, matched_node_ids)
    ]
    edges = tuple(
        sorted(
            candidate_edges,
            key=lambda edge: (
                edge.source_node_id not in matched_node_ids
                and edge.target_node_id not in matched_node_ids,
                -edge.confidence,
                edge.id,
            ),
        )[:max_edges]
    )

    node_by_id = {node.id: node for node in graph.nodes}
    result_nodes = list(matched_nodes)
    seen_node_ids = {node.id for node in result_nodes}
    for edge in edges:
        for node_id in (edge.source_node_id, edge.target_node_id):
            if node_id in seen_node_ids or node_id not in node_by_id:
                continue
            result_nodes.append(node_by_id[node_id])
            seen_node_ids.add(node_id)
            if len(result_nodes) >= max_nodes:
                break

    return GraphSearchResult(
        search_terms=terms,
        relation_types=tuple(relation_types),
        nodes=tuple(result_nodes),
        edges=edges,
    )


def _edge_matches(
    edge: GraphEdge,
    normalized_terms: Sequence[str],
    relation_filter: set[str],
    matched_node_ids: set[str],
) -> bool:
    connected = (
        edge.source_node_id in matched_node_ids
        or edge.target_node_id in matched_node_ids
    )
    relation_matches = not relation_filter or edge.relation_type in relation_filter
    if connected and relation_matches:
        return True

    edge_text = _normalize(
        " ".join(
            [
                edge.relation_type,
                edge.raw_relation,
                edge.description or "",
            ]
        )
    )
    text_matches = any(term and term in edge_text for term in normalized_terms)
    return not matched_node_ids and relation_matches and text_matches


def _node_score(node: GraphNode, normalized_terms: Sequence[str]) -> int:
    if not normalized_terms:
        return 0

    names = [node.name, *node.aliases]
    normalized_names = [_normalize(name) for name in names]
    normalized_description = _normalize(node.description or "")
    score = 0
    for term in normalized_terms:
        if term in normalized_names:
            score = max(score, 100)
        elif any(term in name or name in term for name in normalized_names):
            score = max(score, 80)
        elif term in normalized_description:
            score = max(score, 30)
    return score


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = _normalize(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
