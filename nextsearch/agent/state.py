from __future__ import annotations

from typing import TypedDict

from nextsearch.agent.models import Citation, EvidenceRecord, GraphSearchDecision, QueryPlan
from nextsearch.ingestion.graph.models import RelationType
from nextsearch.retrieval.graph_search import GraphSearchResult


class QueryAgentInput(TypedDict):
    query: str


class QueryAgentOutput(TypedDict):
    answer: str
    citations: list[Citation]
    evidence: list[EvidenceRecord]


class QueryAgentState(QueryAgentInput, total=False):
    plan: QueryPlan
    search_terms: list[str]
    relation_types: list[RelationType]
    graph_results: list[GraphSearchResult]
    evidence: list[EvidenceRecord]
    search_iterations: int
    max_search_iterations: int
    decision: GraphSearchDecision
    answer: str
    citations: list[Citation]
