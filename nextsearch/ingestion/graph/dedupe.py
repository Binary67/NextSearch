from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from nextsearch.ingestion.errors import GraphDedupeError
from nextsearch.ingestion.graph.llm_extractor import normalize_edge_id
from nextsearch.ingestion.graph.models import (
    GraphDedupeResult,
    GraphEdge,
    GraphNode,
    GraphNodeMergeDecision,
    KnowledgeGraph,
    NodeMergeDecisionType,
    SourceRef,
)
from nextsearch.llm.service import LLMService
from nextsearch.llm.types import LLMMessage


MERGE_CONFIDENCE_THRESHOLD = 0.9
TOKEN_SIMILARITY_THRESHOLD = 0.88
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

SYSTEM_PROMPT = """You decide whether two knowledge graph nodes refer to the same real-world entity.
Use only the provided node evidence and neighboring graph context.
Return same only when the evidence supports one entity.
Return uncertain when the evidence is weak or ambiguous.
Do not invent facts or canonical names."""


class NodeMergeAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: NodeMergeDecisionType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = None


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


def dedupe_knowledge_graph(
    graph: KnowledgeGraph,
    llm: LLMService,
) -> GraphDedupeResult:
    node_by_id = {node.id: node for node in graph.nodes}
    candidates = _generate_merge_candidates(graph)
    union_find = _UnionFind(list(node_by_id))
    decisions: list[GraphNodeMergeDecision] = []

    for candidate in candidates:
        if union_find.find(candidate.source_node_id) == union_find.find(
            candidate.target_node_id
        ):
            continue

        adjudication = _adjudicate_candidate(
            graph=graph,
            node_by_id=node_by_id,
            candidate=candidate,
            llm=llm,
        )
        should_merge = (
            adjudication.decision == "same"
            and adjudication.confidence >= MERGE_CONFIDENCE_THRESHOLD
        )
        if should_merge:
            union_find.union(candidate.source_node_id, candidate.target_node_id)

        decisions.append(
            GraphNodeMergeDecision(
                node_ids=[candidate.source_node_id, candidate.target_node_id],
                decision=adjudication.decision,
                confidence=adjudication.confidence,
                reason=adjudication.reason,
            )
        )

    replacements, canonical_by_root = _build_node_replacements(
        graph.nodes,
        union_find,
    )
    cleaned_graph = _apply_node_replacements(
        graph=graph,
        replacements=replacements,
        canonical_by_root=canonical_by_root,
        union_find=union_find,
    )
    decisions = [
        _attach_canonical_node_id(decision, union_find, canonical_by_root)
        for decision in decisions
    ]
    return GraphDedupeResult(
        graph=cleaned_graph,
        merge_decisions=decisions,
        node_id_replacements=replacements,
    )


def _generate_merge_candidates(graph: KnowledgeGraph) -> list[_MergeCandidate]:
    nodes_by_type: dict[str, list[GraphNode]] = defaultdict(list)
    for node in graph.nodes:
        nodes_by_type[node.type].append(node)

    neighbor_ids = _neighbor_ids_by_node(graph.edges)
    candidates: list[_MergeCandidate] = []
    for typed_nodes in nodes_by_type.values():
        for left, right in combinations(typed_nodes, 2):
            reasons = _candidate_reasons(left, right, neighbor_ids)
            if reasons:
                candidates.append(
                    _MergeCandidate(
                        source_node_id=left.id,
                        target_node_id=right.id,
                        reasons=tuple(sorted(reasons)),
                    )
                )

    return sorted(
        candidates,
        key=lambda candidate: (candidate.source_node_id, candidate.target_node_id),
    )


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


def _adjudicate_candidate(
    *,
    graph: KnowledgeGraph,
    node_by_id: dict[str, GraphNode],
    candidate: _MergeCandidate,
    llm: LLMService,
) -> NodeMergeAdjudication:
    try:
        return llm.generate_json(
            role="graph_extraction",
            messages=_build_adjudication_messages(
                graph=graph,
                node_by_id=node_by_id,
                candidate=candidate,
            ),
            response_model=NodeMergeAdjudication,
            temperature=0,
        )
    except Exception as exc:
        raise GraphDedupeError(
            "Graph node dedupe failed for "
            f"{candidate.source_node_id} and {candidate.target_node_id}"
        ) from exc


def _build_adjudication_messages(
    *,
    graph: KnowledgeGraph,
    node_by_id: dict[str, GraphNode],
    candidate: _MergeCandidate,
) -> list[LLMMessage]:
    left = node_by_id[candidate.source_node_id]
    right = node_by_id[candidate.target_node_id]
    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=(
                "Do these two knowledge graph nodes refer to the same real-world "
                "entity?\n"
                f"Candidate reasons: {', '.join(candidate.reasons)}\n\n"
                f"Node A:\n{_node_summary(left, graph, node_by_id)}\n\n"
                f"Node B:\n{_node_summary(right, graph, node_by_id)}\n\n"
                "Return decision same, different, or uncertain."
            ),
        ),
    ]


def _node_summary(
    node: GraphNode,
    graph: KnowledgeGraph,
    node_by_id: dict[str, GraphNode],
) -> str:
    aliases = ", ".join(node.aliases) if node.aliases else "none"
    description = node.description or "none"
    quotes = "\n".join(_source_quote_lines(node.source_refs)) or "- none"
    edges = "\n".join(_neighbor_edge_lines(node, graph.edges, node_by_id)) or "- none"
    return (
        f"id: {node.id}\n"
        f"type: {node.type}\n"
        f"name: {node.name}\n"
        f"aliases: {aliases}\n"
        f"description: {description}\n"
        f"source quotes:\n{quotes}\n"
        f"neighboring edges:\n{edges}"
    )


def _source_quote_lines(source_refs: list[SourceRef]) -> list[str]:
    lines: list[str] = []
    for source_ref in source_refs[:MAX_SOURCE_QUOTES]:
        quote = source_ref.quote.strip()
        if len(quote) > MAX_QUOTE_CHARS:
            quote = quote[:MAX_QUOTE_CHARS].rstrip() + "..."
        lines.append(f"- {quote}")
    return lines


def _neighbor_edge_lines(
    node: GraphNode,
    edges: list[GraphEdge],
    node_by_id: dict[str, GraphNode],
) -> list[str]:
    lines: list[str] = []
    for edge in edges:
        if edge.source_node_id != node.id and edge.target_node_id != node.id:
            continue
        source_name = _node_display_name(edge.source_node_id, node_by_id)
        target_name = _node_display_name(edge.target_node_id, node_by_id)
        lines.append(f"- {source_name} -{edge.relation_type}-> {target_name}")
        if len(lines) == MAX_EDGE_SUMMARIES:
            break
    return lines


def _node_display_name(node_id: str, node_by_id: dict[str, GraphNode]) -> str:
    node = node_by_id.get(node_id)
    if node is None:
        return node_id
    return node.name


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


def _merge_source_refs(
    existing: list[SourceRef],
    incoming: list[SourceRef],
) -> list[SourceRef]:
    merged = list(existing)
    seen = {
        (
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


def _neighbor_ids_by_node(edges: list[GraphEdge]) -> dict[str, set[str]]:
    neighbor_ids: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        neighbor_ids[edge.source_node_id].add(edge.target_node_id)
        neighbor_ids[edge.target_node_id].add(edge.source_node_id)
    return neighbor_ids


def _node_names(node: GraphNode) -> list[str]:
    return [node.name, *node.aliases]


def _normalize_text(value: str) -> str:
    return " ".join(_name_tokens(value))


def _name_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


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


def _has_organization_suffix(value: str) -> bool:
    return any(token in ORGANIZATION_SUFFIX_TOKENS for token in _name_tokens(value))


def _is_acronym_only(value: str) -> bool:
    tokens = _name_tokens(value)
    return len(tokens) == 1 and 2 <= len(tokens[0]) <= 8
