from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nextsearch.ingestion.graph.models import (
    RELATION_TYPES,
    KnowledgeGraph,
    RelationTypeProposal,
    RelationTypeProposalSet,
    SourceRef,
)
from nextsearch.llm.service import LLMService
from nextsearch.llm.types import LLMMessage


PROMOTION_CONFIDENCE_THRESHOLD = 0.85
PROMOTION_EVIDENCE_THRESHOLD = 3
PROMOTION_DOCUMENT_THRESHOLD = 2

SYSTEM_PROMPT = """You review knowledge graph relationship evidence.
Suggest new snake_case relationship types only when raw relationship wording is
more specific and reusable than the closest existing canonical relation type.
Do not suggest labels that duplicate an existing canonical relation type.
Return no proposal when the existing relation type is sufficient."""


class _RelationTypeProposalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[RelationTypeProposal] = Field(default_factory=list)


class _EdgeSupport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str
    relation_type: str
    raw_relation: str
    source_node_type: str
    target_node_type: str
    evidence_count: int
    document_ids: list[str]


def build_relation_type_proposals(
    graph: KnowledgeGraph,
    llm: LLMService,
) -> RelationTypeProposalSet:
    support_by_edge_id = _edge_support_by_id(graph)
    if not support_by_edge_id:
        return RelationTypeProposalSet(
            document_id=graph.document_id,
            content_hash=graph.content_hash,
        )

    response = llm.generate_json(
        role="graph_extraction",
        messages=_build_messages(graph, list(support_by_edge_id.values())),
        response_model=_RelationTypeProposalResponse,
        temperature=0,
    )
    proposals = [
        _normalize_proposal(proposal, support_by_edge_id)
        for proposal in response.proposals
    ]
    return RelationTypeProposalSet(
        document_id=graph.document_id,
        content_hash=graph.content_hash,
        proposals=[proposal for proposal in proposals if proposal is not None],
    )


def _edge_support_by_id(graph: KnowledgeGraph) -> dict[str, _EdgeSupport]:
    node_by_id = {node.id: node for node in graph.nodes}
    support_by_edge_id: dict[str, _EdgeSupport] = {}
    for edge in graph.edges:
        source_node = node_by_id[edge.source_node_id]
        target_node = node_by_id[edge.target_node_id]
        support_by_edge_id[edge.id] = _EdgeSupport(
            edge_id=edge.id,
            relation_type=edge.relation_type,
            raw_relation=edge.raw_relation,
            source_node_type=source_node.type,
            target_node_type=target_node.type,
            evidence_count=len(edge.source_refs),
            document_ids=sorted(_document_ids(edge.source_refs)),
        )
    return support_by_edge_id


def _build_messages(
    graph: KnowledgeGraph,
    edge_supports: list[_EdgeSupport],
) -> list[LLMMessage]:
    canonical_relation_types = ", ".join(RELATION_TYPES)
    edge_lines = "\n".join(
        (
            f"- edge_id: {support.edge_id}\n"
            f"  relation_type: {support.relation_type}\n"
            f"  raw_relation: {support.raw_relation}\n"
            f"  source_node_type: {support.source_node_type}\n"
            f"  target_node_type: {support.target_node_type}\n"
            f"  evidence_count: {support.evidence_count}\n"
            f"  document_ids: {', '.join(support.document_ids)}"
        )
        for support in edge_supports
    )
    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=(
                "Review these extracted graph edges and propose reusable relation "
                "types that are not already covered by the canonical list.\n"
                f"Document id: {graph.document_id}\n"
                f"Canonical relation types: {canonical_relation_types}\n\n"
                "Use only the listed edge_id values as supporting_edge_ids.\n"
                "Set closest_existing_relation_type to the current canonical type "
                "that best fits the evidence.\n\n"
                f"Edges:\n{edge_lines}"
            ),
        ),
    ]


def _normalize_proposal(
    proposal: RelationTypeProposal,
    support_by_edge_id: dict[str, _EdgeSupport],
) -> RelationTypeProposal | None:
    supporting_edge_ids = [
        edge_id
        for edge_id in proposal.supporting_edge_ids
        if edge_id in support_by_edge_id
    ]
    if not supporting_edge_ids:
        return None

    raw_relations = sorted(
        {
            support_by_edge_id[edge_id].raw_relation.strip()
            for edge_id in supporting_edge_ids
            if support_by_edge_id[edge_id].raw_relation.strip()
        }
    )
    source_node_types = sorted(
        {
            support_by_edge_id[edge_id].source_node_type
            for edge_id in supporting_edge_ids
        }
    )
    target_node_types = sorted(
        {
            support_by_edge_id[edge_id].target_node_type
            for edge_id in supporting_edge_ids
        }
    )
    evidence_count = sum(
        support_by_edge_id[edge_id].evidence_count
        for edge_id in supporting_edge_ids
    )
    document_ids = {
        document_id
        for edge_id in supporting_edge_ids
        for document_id in support_by_edge_id[edge_id].document_ids
    }
    if not raw_relations or evidence_count == 0 or not document_ids:
        return None

    promotion_ready = _promotion_ready(
        proposal=proposal,
        evidence_count=evidence_count,
        document_count=len(document_ids),
    )
    return RelationTypeProposal.model_validate(
        {
            **proposal.model_dump(mode="json"),
            "raw_relations": raw_relations,
            "source_node_types": source_node_types,
            "target_node_types": target_node_types,
            "supporting_edge_ids": supporting_edge_ids,
            "evidence_count": evidence_count,
            "document_count": len(document_ids),
            "status": "proposed",
            "promotion_ready": promotion_ready,
        }
    )


def _promotion_ready(
    *,
    proposal: RelationTypeProposal,
    evidence_count: int,
    document_count: int,
) -> bool:
    return (
        proposal.confidence >= PROMOTION_CONFIDENCE_THRESHOLD
        and evidence_count >= PROMOTION_EVIDENCE_THRESHOLD
        and document_count >= PROMOTION_DOCUMENT_THRESHOLD
        and proposal.proposed_relation_type not in RELATION_TYPES
    )


def _document_ids(source_refs: list[SourceRef]) -> set[str]:
    return {source_ref.document_id for source_ref in source_refs}
