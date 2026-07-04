from __future__ import annotations

from typing import Collection

from nextsearch.ingestion.graph._dedupe_adjudication import _adjudicate_candidate
from nextsearch.ingestion.graph._dedupe_candidates import _generate_merge_candidates
from nextsearch.ingestion.graph._dedupe_common import (
    MERGE_CONFIDENCE_THRESHOLD,
    _UnionFind,
)
from nextsearch.ingestion.graph._dedupe_rewrite import (
    _apply_node_replacements,
    _attach_canonical_node_id,
    _build_node_replacements,
)
from nextsearch.ingestion.graph.models import (
    GraphDedupeResult,
    GraphNode,
    GraphNodeMergeDecision,
    KnowledgeGraph,
)
from nextsearch.llm.service import LLMService


def dedupe_knowledge_graph(
    graph: KnowledgeGraph,
    llm: LLMService,
) -> GraphDedupeResult:
    return _dedupe_knowledge_graph(
        graph=graph,
        llm=llm,
        required_node_ids=None,
    )


def dedupe_knowledge_graph_incremental(
    graph: KnowledgeGraph,
    llm: LLMService,
    *,
    incoming_node_ids: Collection[str],
) -> GraphDedupeResult:
    return _dedupe_knowledge_graph(
        graph=graph,
        llm=llm,
        required_node_ids=set(incoming_node_ids),
    )


def _dedupe_knowledge_graph(
    *,
    graph: KnowledgeGraph,
    llm: LLMService,
    required_node_ids: set[str] | None,
) -> GraphDedupeResult:
    node_by_id = {node.id: node for node in graph.nodes}
    candidates = _generate_merge_candidates(
        graph,
        llm=llm,
        required_node_ids=required_node_ids,
    )
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
