from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Sequence

from nextsearch.ingestion.graph.embeddings import (
    GraphEmbeddingIndex,
    GraphEmbeddingIndexError,
    validate_graph_embedding_index,
)
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
    query_embeddings: Sequence[Sequence[float]],
    graph_embeddings: GraphEmbeddingIndex,
    embedding_provider: str,
    embedding_model: str,
    relation_types: Sequence[str] = (),
    max_nodes: int = 8,
    max_edges: int = 16,
) -> GraphSearchResult:
    terms = tuple(_dedupe(term.strip() for term in search_terms if term.strip()))
    if len(query_embeddings) != len(terms):
        raise GraphEmbeddingIndexError(
            "Graph query embedding count does not match search term count"
        )
    normalized_terms = tuple(_normalize(term) for term in terms)
    relation_filter = set(relation_types)
    embedding_items = validate_graph_embedding_index(
        graph,
        graph_embeddings,
        provider=embedding_provider,
        model=embedding_model,
    )
    query_vectors = [list(embedding) for embedding in query_embeddings]
    node_embedding_by_id = {
        item_id: item.embedding
        for (item_type, item_id), item in embedding_items.items()
        if item_type == "node"
    }
    edge_embedding_by_id = {
        item_id: item.embedding
        for (item_type, item_id), item in embedding_items.items()
        if item_type == "edge"
    }

    node_scores = [
        (_node_rank(node, normalized_terms, query_vectors, node_embedding_by_id[node.id]), node)
        for node in graph.nodes
    ]
    matched_nodes = [
        node
        for rank, node in sorted(
            node_scores,
            key=lambda item: (
                -item[0].combined,
                -item[0].semantic_score,
                -item[0].keyword_score,
                item[1].type,
                item[1].name.lower(),
            ),
        )
        if rank.has_signal
    ][:max_nodes]
    matched_node_ids = {node.id for node in matched_nodes}

    candidate_edges = [
        (_edge_rank(
            edge,
            normalized_terms,
            query_vectors,
            edge_embedding_by_id[edge.id],
            relation_filter,
            matched_node_ids,
        ), edge)
        for edge in graph.edges
    ]
    edges = tuple(
        edge
        for rank, edge in sorted(
            (
                (rank, edge)
                for rank, edge in candidate_edges
                if rank.has_signal
            ),
            key=lambda item: (
                -item[0].combined,
                -item[0].semantic_score,
                -item[0].keyword_score,
                not item[0].connected,
                -item[1].confidence,
                item[1].id,
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


@dataclass(frozen=True)
class _NodeRank:
    semantic_score: float
    keyword_score: int

    @property
    def combined(self) -> float:
        return self.semantic_score + (self.keyword_score / 100.0)

    @property
    def has_signal(self) -> bool:
        return self.semantic_score > 0.0 or self.keyword_score > 0


@dataclass(frozen=True)
class _EdgeRank:
    semantic_score: float
    keyword_score: int
    connected: bool
    relation_matches: bool

    @property
    def combined(self) -> float:
        connected_boost = 0.25 if self.connected else 0.0
        return self.semantic_score + (self.keyword_score / 100.0) + connected_boost

    @property
    def has_signal(self) -> bool:
        return self.relation_matches and (
            self.semantic_score > 0.0
            or self.keyword_score > 0
            or self.connected
        )


def _node_rank(
    node: GraphNode,
    normalized_terms: Sequence[str],
    query_vectors: Sequence[list[float]],
    embedding: list[float],
) -> _NodeRank:
    return _NodeRank(
        semantic_score=_max_cosine_similarity(query_vectors, embedding),
        keyword_score=_node_score(node, normalized_terms),
    )


def _edge_rank(
    edge: GraphEdge,
    normalized_terms: Sequence[str],
    query_vectors: Sequence[list[float]],
    embedding: list[float],
    relation_filter: set[str],
    matched_node_ids: set[str],
) -> _EdgeRank:
    connected = (
        edge.source_node_id in matched_node_ids
        or edge.target_node_id in matched_node_ids
    )
    relation_matches = not relation_filter or edge.relation_type in relation_filter
    return _EdgeRank(
        semantic_score=_max_cosine_similarity(query_vectors, embedding),
        keyword_score=_edge_keyword_score(edge, normalized_terms),
        connected=connected,
        relation_matches=relation_matches,
    )


def _edge_keyword_score(edge: GraphEdge, normalized_terms: Sequence[str]) -> int:
    edge_text = _normalize(
        " ".join(
            [
                edge.relation_type,
                edge.raw_relation,
                edge.description or "",
            ]
        )
    )
    score = 0
    normalized_relation_type = _normalize(edge.relation_type)
    for term in normalized_terms:
        if term == normalized_relation_type:
            score = max(score, 100)
        elif term and term in edge_text:
            score = max(score, 50)
    return score


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


def _max_cosine_similarity(
    query_vectors: Sequence[list[float]],
    embedding: list[float],
) -> float:
    return max(
        (_cosine_similarity(query_vector, embedding) for query_vector in query_vectors),
        default=0.0,
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0

    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    ) / (left_norm * right_norm)


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
