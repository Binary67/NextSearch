from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ENTITY_TYPES = (
    "person",
    "organization",
    "location",
    "project",
    "product",
    "system",
    "document",
    "concept",
    "requirement",
    "risk",
    "event",
    "metric",
    "date",
    "other",
)

RELATION_TYPES = (
    "mentions",
    "related_to",
    "part_of",
    "owned_by",
    "created_by",
    "uses",
    "depends_on",
    "impacts",
    "causes",
    "supports",
    "contradicts",
    "requires",
    "located_in",
    "has_risk",
    "has_requirement",
    "has_metric",
    "happened_on",
)

EntityType = Literal[
    "person",
    "organization",
    "location",
    "project",
    "product",
    "system",
    "document",
    "concept",
    "requirement",
    "risk",
    "event",
    "metric",
    "date",
    "other",
]

RelationType = Literal[
    "mentions",
    "related_to",
    "part_of",
    "owned_by",
    "created_by",
    "uses",
    "depends_on",
    "impacts",
    "causes",
    "supports",
    "contradicts",
    "requires",
    "located_in",
    "has_risk",
    "has_requirement",
    "has_metric",
    "happened_on",
]


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    heading: str
    page_start: int | None = None
    page_end: int | None = None
    quote: str = Field(min_length=1)


class NodeRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: EntityType
    name: str = Field(min_length=1)


class ExtractedNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: EntityType
    name: str = Field(min_length=1)
    description: str | None = None
    source_refs: list[SourceRef] = Field(min_length=1)


class ExtractedEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: NodeRef
    target: NodeRef
    relation_type: RelationType
    raw_relation: str = Field(min_length=1)
    description: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_refs: list[SourceRef] = Field(min_length=1)


class SectionGraphExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[ExtractedNode] = Field(default_factory=list)
    edges: list[ExtractedEdge] = Field(default_factory=list)


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: EntityType
    name: str
    description: str | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_node_id: str
    target_node_id: str
    relation_type: RelationType
    raw_relation: str
    description: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_refs: list[SourceRef] = Field(default_factory=list)


class KnowledgeGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    source_path: str
    page_count: int
    nodes: list[GraphNode]
    edges: list[GraphEdge]
