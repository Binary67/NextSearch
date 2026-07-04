from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nextsearch.ingestion.artifacts import (
    read_graph_artifact,
    read_graph_embedding_artifact,
)
from nextsearch.ingestion.graph.embeddings import (
    GraphEmbeddingIndex,
    GraphEmbeddingIndexError,
)
from nextsearch.ingestion.graph.models import KnowledgeGraph


class GraphStore(Protocol):
    def load_graph(self) -> KnowledgeGraph:
        ...

    def load_graph_embeddings(self) -> GraphEmbeddingIndex:
        ...


@dataclass(frozen=True)
class JsonGraphStore:
    graph_path: Path
    embedding_path: Path

    @classmethod
    def from_artifact_dir(cls, artifact_dir: Path) -> JsonGraphStore:
        path = Path(artifact_dir)
        return cls(path / "graph.json", path / "graph_embeddings.json")

    def load_graph(self) -> KnowledgeGraph:
        return read_graph_artifact(self.graph_path)

    def load_graph_embeddings(self) -> GraphEmbeddingIndex:
        try:
            return read_graph_embedding_artifact(self.embedding_path)
        except FileNotFoundError as exc:
            raise GraphEmbeddingIndexError(
                f"Graph embedding index not found: {self.embedding_path}"
            ) from exc
