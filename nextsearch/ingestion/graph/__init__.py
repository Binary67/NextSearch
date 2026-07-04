"""Knowledge graph extraction helpers."""

from nextsearch.ingestion.graph.dedupe import (
    dedupe_knowledge_graph,
    dedupe_knowledge_graph_incremental,
)
from nextsearch.ingestion.graph.llm_extractor import (
    extract_knowledge_graph_from_markdown,
)
from nextsearch.ingestion.graph.merge import (
    merge_knowledge_graphs,
    remove_document_from_graph,
)
from nextsearch.ingestion.graph.models import (
    ExtractedSourceRef,
    GraphDedupeResult,
    GraphEdge,
    GraphNode,
    GraphNodeMergeDecision,
    KnowledgeGraph,
    RelationTypeProposal,
    RelationTypeProposalSet,
    SourceRef,
)
from nextsearch.ingestion.graph.relation_proposals import (
    build_relation_type_proposals,
)

__all__ = [
    "GraphDedupeResult",
    "GraphEdge",
    "GraphNode",
    "GraphNodeMergeDecision",
    "KnowledgeGraph",
    "RelationTypeProposal",
    "RelationTypeProposalSet",
    "SourceRef",
    "ExtractedSourceRef",
    "build_relation_type_proposals",
    "dedupe_knowledge_graph",
    "dedupe_knowledge_graph_incremental",
    "extract_knowledge_graph_from_markdown",
    "merge_knowledge_graphs",
    "remove_document_from_graph",
]
