from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nextsearch.ingestion.artifacts import read_graph_artifact
from nextsearch.ingestion.graph.models import KnowledgeGraph


class GraphStore(Protocol):
    def load_graph(self) -> KnowledgeGraph:
        ...


@dataclass(frozen=True)
class JsonGraphStore:
    graph_path: Path

    @classmethod
    def from_artifact_dir(cls, artifact_dir: Path) -> JsonGraphStore:
        return cls(Path(artifact_dir) / "graph.json")

    def load_graph(self) -> KnowledgeGraph:
        return read_graph_artifact(self.graph_path)
