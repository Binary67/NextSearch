"""Query-time retrieval helpers for NextSearch."""

from nextsearch.retrieval.evidence import build_evidence_records
from nextsearch.ingestion.graph.embeddings import GraphEmbeddingIndexError
from nextsearch.retrieval.graph_search import GraphSearchResult, search_graph
from nextsearch.retrieval.graph_store import GraphStore, JsonGraphStore
from nextsearch.retrieval.source_store import MarkdownArtifactSourceStore, SourceStore

__all__ = [
    "GraphEmbeddingIndexError",
    "GraphSearchResult",
    "GraphStore",
    "JsonGraphStore",
    "MarkdownArtifactSourceStore",
    "SourceStore",
    "build_evidence_records",
    "search_graph",
]
