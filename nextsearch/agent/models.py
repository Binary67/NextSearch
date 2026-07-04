from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nextsearch.ingestion.graph.models import RelationType


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_terms: list[str] = Field(default_factory=list)
    relation_types: list[RelationType] = Field(default_factory=list)
    rationale: str | None = None


class GraphSearchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    next_step: Literal["search_more", "answer"]
    search_terms: list[str] = Field(default_factory=list)
    relation_types: list[RelationType] = Field(default_factory=list)
    reason: str | None = None


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    section_id: str
    heading: str
    page_start: int | None = None
    page_end: int | None = None
    quote: str


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: int
    citation: Citation
    snippet: str
    graph_node_ids: list[str] = Field(default_factory=list)
    graph_edge_ids: list[str] = Field(default_factory=list)


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    cited_evidence_ids: list[int] = Field(default_factory=list)


class AgentAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
