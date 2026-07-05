from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Iterable, Sequence

from nextsearch.agent.models import (
    AnswerDraft,
    Citation,
    EvidenceRecord,
    GraphSearchDecision,
    QueryPlan,
    QueryRequest,
)
from nextsearch.agent.prompts import (
    build_answer_messages,
    build_query_planning_messages,
    build_search_decision_messages,
)
from nextsearch.agent.state import QueryAgentState
from nextsearch.llm.service import LLMService
from nextsearch.retrieval.evidence import build_evidence_records
from nextsearch.retrieval.graph_search import search_graph
from nextsearch.retrieval.graph_store import GraphStore
from nextsearch.retrieval.source_store import SourceStore


@dataclass(frozen=True)
class QueryAgentNodes:
    llm: LLMService
    graph_store: GraphStore
    source_store: SourceStore
    max_search_iterations: int = 3
    max_graph_nodes: int = 8
    max_graph_edges: int = 16

    def plan_query(self, state: QueryAgentState) -> dict[str, object]:
        request = QueryRequest.model_validate({"query": state["query"]})
        plan = self.llm.generate_json(
            role="query_planning",
            messages=build_query_planning_messages(request.query),
            response_model=QueryPlan,
            temperature=0,
        )
        search_terms = _prepare_search_terms(plan.search_terms, request.query)
        return {
            "plan": plan,
            "search_terms": search_terms,
            "relation_types": plan.relation_types,
            "search_iterations": 0,
            "max_search_iterations": self.max_search_iterations,
        }

    def search_graph(self, state: QueryAgentState) -> dict[str, object]:
        graph = self.graph_store.load_graph()
        search_terms = _prepare_search_terms(
            state.get("search_terms", []),
            state["query"],
        )
        query_embeddings = self.llm.embed(
            role="graph_query_embedding",
            texts=search_terms,
        ).embeddings
        result = search_graph(
            graph,
            search_terms=search_terms,
            query_embeddings=query_embeddings,
            graph_embeddings=self.graph_store.load_graph_embeddings(),
            embedding_provider=self.llm.embedding_provider_name(),
            embedding_model=self.llm.embedding_model(),
            relation_types=state.get("relation_types", []),
            max_nodes=self.max_graph_nodes,
            max_edges=self.max_graph_edges,
        )
        return {
            "graph_results": [*state.get("graph_results", []), result],
            "search_iterations": state.get("search_iterations", 0) + 1,
        }

    def inspect_evidence(self, state: QueryAgentState) -> dict[str, object]:
        results = state.get("graph_results", [])
        if not results:
            return {"evidence": state.get("evidence", [])}

        candidates = build_evidence_records(
            results[-1],
            self.source_store,
            start_id=1,
        )
        return {
            "evidence": _merge_evidence(state.get("evidence", []), candidates),
        }

    def decide_next_step(self, state: QueryAgentState) -> dict[str, object]:
        search_iterations = state.get("search_iterations", 0)
        max_search_iterations = state.get(
            "max_search_iterations",
            self.max_search_iterations,
        )
        if search_iterations >= max_search_iterations:
            return {
                "decision": GraphSearchDecision(
                    next_step="answer",
                    reason="Maximum graph search iterations reached.",
                )
            }

        decision = self.llm.generate_json(
            role="graph_search_decision",
            messages=build_search_decision_messages(
                query=state["query"],
                plan=state["plan"],
                graph_results=state.get("graph_results", []),
                evidence=state.get("evidence", []),
                search_iterations=search_iterations,
                max_search_iterations=max_search_iterations,
            ),
            response_model=GraphSearchDecision,
            temperature=0,
        )
        if decision.next_step == "search_more" and decision.search_terms:
            return {
                "decision": decision,
                "search_terms": decision.search_terms,
                "relation_types": decision.relation_types,
            }

        return {
            "decision": GraphSearchDecision(
                next_step="answer",
                reason=decision.reason,
            )
        }

    def generate_answer(self, state: QueryAgentState) -> dict[str, object]:
        evidence = state.get("evidence", [])
        if not evidence:
            return {
                "answer": (
                    "I do not have enough source evidence to answer this query."
                ),
                "citations": [],
                "evidence": [],
            }

        draft = self.llm.generate_json(
            role="answer_generation",
            messages=build_answer_messages(
                query=state["query"],
                evidence=evidence,
            ),
            response_model=AnswerDraft,
            temperature=0,
        )
        evidence_by_id = {record.evidence_id: record for record in evidence}
        cited_records = [
            evidence_by_id[evidence_id]
            for evidence_id in draft.cited_evidence_ids
            if evidence_id in evidence_by_id
        ]
        return {
            "answer": draft.answer,
            "citations": _dedupe_citations(
                record.citation for record in cited_records
            ),
            "evidence": evidence,
        }


def route_after_decision(state: QueryAgentState) -> str:
    decision = state.get("decision")
    if decision is not None and decision.next_step == "search_more":
        return "search_graph"
    return "generate_answer"


def _merge_evidence(
    existing: Sequence[EvidenceRecord],
    candidates: Sequence[EvidenceRecord],
) -> list[EvidenceRecord]:
    merged = list(existing)
    record_by_key = {_evidence_key(record): record for record in merged}
    for candidate in candidates:
        key = _evidence_key(candidate)
        record = record_by_key.get(key)
        if record is None:
            record = candidate.model_copy(update={"evidence_id": len(merged) + 1})
            merged.append(record)
            record_by_key[key] = record
            continue

        for node_id in candidate.graph_node_ids:
            if node_id not in record.graph_node_ids:
                record.graph_node_ids.append(node_id)
        for edge_id in candidate.graph_edge_ids:
            if edge_id not in record.graph_edge_ids:
                record.graph_edge_ids.append(edge_id)

    return merged


def _evidence_key(record: EvidenceRecord) -> tuple[str, str, str, str]:
    return (
        record.citation.document_id,
        record.citation.section_id,
        record.citation.quote,
        record.snippet,
    )


def _dedupe_citations(citations: Iterable[Citation]) -> list[Citation]:
    result: list[Citation] = []
    seen: set[tuple[str, str, str]] = set()
    for citation in citations:
        key = (
            citation.document_id,
            citation.section_id,
            citation.quote,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(citation)
    return result


def _prepare_search_terms(search_terms: Sequence[str], fallback_query: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for term in search_terms:
        value = term.strip()
        key = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
    if result:
        return result
    return [fallback_query.strip() or fallback_query]
