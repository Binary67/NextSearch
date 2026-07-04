from __future__ import annotations

from collections.abc import Sequence

from nextsearch.agent.models import EvidenceRecord, QueryPlan
from nextsearch.llm.types import LLMMessage
from nextsearch.retrieval.graph_search import GraphSearchResult


QUERY_PLANNING_SYSTEM_PROMPT = """You plan knowledge graph searches for a grounded
document assistant.
Return focused entity, concept, or relationship terms from the user query.
Prefer exact names from the query over broad keywords."""

SEARCH_DECISION_SYSTEM_PROMPT = """You decide whether a grounded document assistant
needs another knowledge graph search.
Answer only when the available source evidence is enough to address the user query.
Search again when a specific missing entity, concept, or relationship would improve the answer."""

ANSWER_SYSTEM_PROMPT = """You answer using only the provided source evidence.
Do not use unsupported graph facts or outside knowledge.
If the evidence is insufficient, say what cannot be determined from the available sources."""


def build_query_planning_messages(query: str) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=QUERY_PLANNING_SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=f"User query:\n{query}",
        ),
    ]


def build_search_decision_messages(
    *,
    query: str,
    plan: QueryPlan,
    graph_results: Sequence[GraphSearchResult],
    evidence: Sequence[EvidenceRecord],
    search_iterations: int,
    max_search_iterations: int,
) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=SEARCH_DECISION_SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=(
                f"User query:\n{query}\n\n"
                f"Initial plan:\n{plan.model_dump(mode='json')}\n\n"
                f"Search iterations: {search_iterations} of {max_search_iterations}\n\n"
                f"Graph search results:\n{format_graph_results(graph_results)}\n\n"
                f"Source evidence:\n{format_evidence(evidence)}"
            ),
        ),
    ]


def build_answer_messages(
    *,
    query: str,
    evidence: Sequence[EvidenceRecord],
) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=(
                f"User query:\n{query}\n\n"
                "Use these evidence IDs for citations. Return cited evidence IDs only "
                "when the answer actually depends on that evidence.\n\n"
                f"{format_evidence(evidence)}"
            ),
        ),
    ]


def format_graph_results(results: Sequence[GraphSearchResult]) -> str:
    if not results:
        return "- none"

    sections: list[str] = []
    for index, result in enumerate(results, start=1):
        node_lines = [
            f"- {node.id}: {node.name} ({node.type})"
            for node in result.nodes
        ] or ["- none"]
        edge_lines = [
            (
                f"- {edge.id}: {edge.source_node_id} "
                f"-[{edge.relation_type}]-> {edge.target_node_id}; "
                f"{edge.raw_relation}"
            )
            for edge in result.edges
        ] or ["- none"]
        sections.append(
            "\n".join(
                [
                    f"Search {index}: {', '.join(result.search_terms) or 'no terms'}",
                    "Nodes:",
                    *node_lines,
                    "Edges:",
                    *edge_lines,
                ]
            )
        )
    return "\n\n".join(sections)


def format_evidence(evidence: Sequence[EvidenceRecord]) -> str:
    if not evidence:
        return "- none"

    return "\n\n".join(
        (
            f"[{record.evidence_id}] "
            f"document={record.citation.document_id}, "
            f"section={record.citation.section_id}, "
            f"heading={record.citation.heading}, "
            f"pages={_page_range(record.citation.page_start, record.citation.page_end)}\n"
            f"Quote: {record.citation.quote}\n"
            f"Snippet:\n{record.snippet}"
        )
        for record in evidence
    )


def _page_range(page_start: int | None, page_end: int | None) -> str:
    if page_start is None and page_end is None:
        return "unknown"
    if page_start == page_end:
        return str(page_start)
    return f"{page_start}-{page_end}"
