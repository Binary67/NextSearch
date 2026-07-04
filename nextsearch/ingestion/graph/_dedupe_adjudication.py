from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nextsearch.ingestion.errors import GraphDedupeError
from nextsearch.ingestion.graph._dedupe_common import (
    MAX_EDGE_SUMMARIES,
    _MergeCandidate,
    _source_quote_lines,
)
from nextsearch.ingestion.graph.models import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    NodeMergeDecisionType,
)
from nextsearch.llm.service import LLMService
from nextsearch.llm.types import LLMMessage


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
