from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nextsearch.ingestion.graph.models import (
    GraphDedupeResult,
    KnowledgeGraph,
    RelationTypeProposalSet,
)
from nextsearch.ingestion.graph.embeddings import (
    GraphEmbeddingIndex,
    build_graph_embedding_index,
)
from nextsearch.ingestion.models import MarkdownDocument
from nextsearch.llm.service import LLMService


def write_markdown_artifacts(document: MarkdownDocument, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = output_dir / "batches"
    batches_dir.mkdir(exist_ok=True)

    (output_dir / "document.md").write_text(document.markdown, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(_manifest(document), indent=2) + "\n",
        encoding="utf-8",
    )

    for index, batch in enumerate(document.batches, start=1):
        batch_path = batches_dir / f"batch-{index:04d}.output.md"
        batch_path.write_text(batch.markdown, encoding="utf-8")


def write_graph_artifact(graph: KnowledgeGraph, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graph.json").write_text(
        json.dumps(graph.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def read_graph_artifact(graph_path: Path) -> KnowledgeGraph:
    return KnowledgeGraph.model_validate_json(graph_path.read_text(encoding="utf-8"))


def write_graph_embedding_artifact(
    graph: KnowledgeGraph,
    llm: LLMService,
    output_dir: Path,
) -> GraphEmbeddingIndex:
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "graph_embeddings.json"
    previous_index = (
        read_graph_embedding_artifact(embedding_path)
        if embedding_path.exists()
        else None
    )
    index = build_graph_embedding_index(
        graph,
        llm,
        previous_index=previous_index,
    )
    embedding_path.write_text(
        json.dumps(index.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return index


def read_graph_embedding_artifact(embedding_path: Path) -> GraphEmbeddingIndex:
    return GraphEmbeddingIndex.model_validate_json(
        embedding_path.read_text(encoding="utf-8")
    )


def write_graph_merge_decisions_artifact(
    result: GraphDedupeResult,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "node_id_replacements": result.node_id_replacements,
        "merge_decisions": [
            decision.model_dump(mode="json")
            for decision in result.merge_decisions
        ],
    }
    (output_dir / "graph_merge_decisions.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def write_relation_type_proposals_artifact(
    proposals: RelationTypeProposalSet,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "relation_type_proposals.json").write_text(
        json.dumps(proposals.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def _manifest(document: MarkdownDocument) -> dict[str, Any]:
    return {
        "source_path": str(document.source_path),
        "page_count": document.page_count,
        "batches": [
            {
                "index": index,
                "page_start": batch.page_start,
                "page_end": batch.page_end,
                "usage": batch.usage,
            }
            for index, batch in enumerate(document.batches, start=1)
        ],
    }
