from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations

from nextsearch.ingestion.errors import GraphDedupeError
from nextsearch.ingestion.graph._dedupe_common import (
    MAX_SEMANTIC_MATCHES_PER_NODE,
    ORGANIZATION_SUFFIX_TOKENS,
    SEMANTIC_SIMILARITY_THRESHOLD,
    TOKEN_SIMILARITY_THRESHOLD,
    _MergeCandidate,
    _candidate_key,
    _name_tokens,
    _node_names,
    _normalize_text,
    _source_quote_lines,
)
from nextsearch.ingestion.graph.models import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
)
from nextsearch.llm.service import LLMService


def _generate_merge_candidates(
    graph: KnowledgeGraph,
    *,
    llm: LLMService,
    required_node_ids: set[str] | None = None,
) -> list[_MergeCandidate]:
    nodes_by_type: dict[str, list[GraphNode]] = defaultdict(list)
    for node in graph.nodes:
        nodes_by_type[node.type].append(node)

    neighbor_ids = _neighbor_ids_by_node(graph.edges)
    candidate_reasons: dict[tuple[str, str], set[str]] = defaultdict(set)
    for typed_nodes in nodes_by_type.values():
        for left, right in combinations(typed_nodes, 2):
            if (
                required_node_ids is not None
                and left.id not in required_node_ids
                and right.id not in required_node_ids
            ):
                continue
            reasons = _candidate_reasons(left, right, neighbor_ids)
            if reasons:
                candidate_reasons[_candidate_key(left.id, right.id)].update(reasons)

        for left_id, right_id in _semantic_candidate_pairs(
            typed_nodes,
            llm=llm,
            required_node_ids=required_node_ids,
        ):
            candidate_reasons[_candidate_key(left_id, right_id)].add(
                "semantic_similarity"
            )

    return sorted(
        [
            _MergeCandidate(
                source_node_id=left_id,
                target_node_id=right_id,
                reasons=tuple(sorted(reasons)),
            )
            for (left_id, right_id), reasons in candidate_reasons.items()
        ],
        key=lambda candidate: (candidate.source_node_id, candidate.target_node_id),
    )


def _semantic_candidate_pairs(
    typed_nodes: list[GraphNode],
    *,
    llm: LLMService,
    required_node_ids: set[str] | None,
) -> set[tuple[str, str]]:
    if len(typed_nodes) < 2:
        return set()

    try:
        embeddings = llm.embed(
            role="document_embedding",
            texts=[_node_embedding_text(node) for node in typed_nodes],
        ).embeddings
    except Exception as exc:
        raise GraphDedupeError("Graph semantic dedupe embedding failed") from exc

    if len(embeddings) != len(typed_nodes):
        raise GraphDedupeError("Graph semantic dedupe returned wrong embedding count")

    matches_by_node_id: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for left_index, right_index in combinations(range(len(typed_nodes)), 2):
        left = typed_nodes[left_index]
        right = typed_nodes[right_index]
        if (
            required_node_ids is not None
            and left.id not in required_node_ids
            and right.id not in required_node_ids
        ):
            continue

        similarity = _cosine_similarity(
            embeddings[left_index],
            embeddings[right_index],
        )
        if similarity < SEMANTIC_SIMILARITY_THRESHOLD:
            continue

        matches_by_node_id[left.id].append((similarity, right.id))
        matches_by_node_id[right.id].append((similarity, left.id))

    top_matches_by_node_id: dict[str, set[str]] = {}
    for node_id, matches in matches_by_node_id.items():
        top_matches_by_node_id[node_id] = {
            other_node_id
            for _similarity, other_node_id in sorted(
                matches,
                key=lambda match: (-match[0], match[1]),
            )[:MAX_SEMANTIC_MATCHES_PER_NODE]
        }

    pairs: set[tuple[str, str]] = set()
    for node_id, other_node_ids in top_matches_by_node_id.items():
        for other_node_id in other_node_ids:
            if node_id not in top_matches_by_node_id.get(other_node_id, set()):
                continue
            pairs.add(_candidate_key(node_id, other_node_id))

    return pairs


def _candidate_reasons(
    left: GraphNode,
    right: GraphNode,
    neighbor_ids: dict[str, set[str]],
) -> set[str]:
    reasons: set[str] = set()
    for left_name in _node_names(left):
        for right_name in _node_names(right):
            left_text = _normalize_text(left_name)
            right_text = _normalize_text(right_name)
            left_tokens = _name_tokens(left_name)
            right_tokens = _name_tokens(right_name)
            if left_text and left_text == right_text:
                reasons.add("normalized_name")
            if left_tokens and set(left_tokens) == set(right_tokens):
                reasons.add("same_tokens")
            left_organization_text = _organization_base_text(left_name)
            right_organization_text = _organization_base_text(right_name)
            if (
                left.type == "organization"
                and left_organization_text
                and left_organization_text == right_organization_text
            ):
                reasons.add("organization_suffix")
            if _is_acronym_match(left_name, right_name, left.type):
                reasons.add("acronym")
            if _token_similarity(left_name, right_name) >= TOKEN_SIMILARITY_THRESHOLD:
                reasons.add("token_similarity")

    shared_neighbors = neighbor_ids.get(left.id, set()) & neighbor_ids.get(
        right.id,
        set(),
    )
    if shared_neighbors and _has_name_overlap(left, right):
        reasons.add("shared_neighbors")

    return {reason for reason in reasons if reason}


def _neighbor_ids_by_node(edges: list[GraphEdge]) -> dict[str, set[str]]:
    neighbor_ids: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        neighbor_ids[edge.source_node_id].add(edge.target_node_id)
        neighbor_ids[edge.target_node_id].add(edge.source_node_id)
    return neighbor_ids


def _organization_base_text(value: str) -> str:
    return " ".join(
        token for token in _name_tokens(value) if token not in ORGANIZATION_SUFFIX_TOKENS
    )


def _is_acronym_match(left: str, right: str, node_type: str) -> bool:
    left_tokens = _base_tokens(left, node_type)
    right_tokens = _base_tokens(right, node_type)
    left_compact = "".join(left_tokens)
    right_compact = "".join(right_tokens)
    left_acronym = "".join(token[0] for token in left_tokens if token)
    right_acronym = "".join(token[0] for token in right_tokens if token)
    return (
        2 <= len(left_compact) <= 8
        and left_compact == right_acronym
        and left_compact != right_compact
    ) or (
        2 <= len(right_compact) <= 8
        and right_compact == left_acronym
        and left_compact != right_compact
    )


def _base_tokens(value: str, node_type: str) -> list[str]:
    if node_type == "organization":
        return [
            token
            for token in _name_tokens(value)
            if token not in ORGANIZATION_SUFFIX_TOKENS
        ]
    return _name_tokens(value)


def _token_similarity(left: str, right: str) -> float:
    left_text = _normalize_text(left)
    right_text = _normalize_text(right)
    if not left_text or not right_text:
        return 0.0
    return max(
        SequenceMatcher(None, left_text, right_text).ratio(),
        SequenceMatcher(
            None,
            " ".join(sorted(left_text.split())),
            " ".join(sorted(right_text.split())),
        ).ratio(),
    )


def _node_embedding_text(node: GraphNode) -> str:
    aliases = ", ".join(node.aliases) if node.aliases else "none"
    description = node.description or "none"
    quotes = "\n".join(_source_quote_lines(node.source_refs)) or "- none"
    return (
        f"type: {node.type}\n"
        f"name: {node.name}\n"
        f"aliases: {aliases}\n"
        f"description: {description}\n"
        f"source quotes:\n{quotes}"
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


def _has_name_overlap(left: GraphNode, right: GraphNode) -> bool:
    left_tokens = {
        token
        for name in _node_names(left)
        for token in _name_tokens(name)
        if token not in ORGANIZATION_SUFFIX_TOKENS
    }
    right_tokens = {
        token
        for name in _node_names(right)
        for token in _name_tokens(name)
        if token not in ORGANIZATION_SUFFIX_TOKENS
    }
    return bool(left_tokens & right_tokens)
