from __future__ import annotations

import hashlib
from pathlib import Path

from nextsearch.ingestion.artifacts import (
    read_graph_artifact,
    write_graph_embedding_artifact,
    write_graph_artifact,
    write_graph_merge_decisions_artifact,
    write_relation_type_proposals_artifact,
    write_markdown_artifacts,
)
from nextsearch.ingestion.graph.dedupe import dedupe_knowledge_graph_incremental
from nextsearch.ingestion.graph.llm_extractor import extract_knowledge_graph_from_markdown
from nextsearch.ingestion.graph.merge import (
    empty_corpus_graph,
    merge_knowledge_graphs,
    remove_document_from_graph,
)
from nextsearch.ingestion.graph.models import KnowledgeGraph
from nextsearch.ingestion.graph.relation_proposals import build_relation_type_proposals
from nextsearch.ingestion.markdown.llm_extractor import extract_markdown
from nextsearch.ingestion.models import MarkdownDocument
from nextsearch.ingestion.sources.pdf import extract_pdf_pages
from nextsearch.llm.service import LLMService


def extract_pdf_to_markdown(
    pdf_path: Path,
    llm: LLMService,
    *,
    output_dir: Path | None = None,
) -> MarkdownDocument:
    source_path = Path(pdf_path)
    pages = extract_pdf_pages(source_path)
    document = extract_markdown(
        pages=pages,
        llm=llm,
        source_path=source_path,
    )

    if output_dir is not None:
        write_markdown_artifacts(document, Path(output_dir))

    return document


def extract_pdf_to_knowledge_graph(
    pdf_path: Path,
    llm: LLMService,
    *,
    document_id: str,
    output_dir: Path | None = None,
) -> KnowledgeGraph:
    source_path = Path(pdf_path)
    content_hash = _file_sha256(source_path)
    document = extract_pdf_to_markdown(pdf_path, llm, output_dir=output_dir)
    graph = extract_knowledge_graph_from_markdown(
        document,
        llm,
        document_id=document_id,
        content_hash=content_hash,
    )

    if output_dir is not None:
        graph_output_dir = Path(output_dir)
        write_graph_artifact(graph, graph_output_dir)
        write_graph_embedding_artifact(graph, llm, graph_output_dir)
        write_relation_type_proposals_artifact(
            build_relation_type_proposals(graph, llm),
            graph_output_dir,
        )

    return graph


def ingest_pdf_to_corpus_graph(
    *,
    pdf_path: Path,
    document_id: str,
    llm: LLMService,
    corpus_graph: KnowledgeGraph | None,
    output_dir: Path | None = None,
) -> KnowledgeGraph:
    source_path = Path(pdf_path)
    content_hash = _file_sha256(source_path)
    output_root = Path(output_dir) if output_dir is not None else None

    if (
        corpus_graph is not None
        and output_root is not None
        and _stored_document_hash(output_root, document_id) == content_hash
    ):
        return corpus_graph

    document_output_dir = (
        output_root / "documents" / document_id
        if output_root is not None
        else None
    )
    document = extract_pdf_to_markdown(
        source_path,
        llm,
        output_dir=document_output_dir,
    )
    document_graph = extract_knowledge_graph_from_markdown(
        document,
        llm,
        document_id=document_id,
        content_hash=content_hash,
    )

    if document_output_dir is not None:
        write_graph_artifact(document_graph, document_output_dir)
        write_graph_embedding_artifact(document_graph, llm, document_output_dir)
        write_relation_type_proposals_artifact(
            build_relation_type_proposals(document_graph, llm),
            document_output_dir,
        )

    base_graph = remove_document_from_graph(
        corpus_graph or empty_corpus_graph(),
        document_id,
    )
    merged_graph = merge_knowledge_graphs(base_graph, document_graph)
    dedupe_result = dedupe_knowledge_graph_incremental(
        merged_graph,
        llm,
        incoming_node_ids={node.id for node in document_graph.nodes},
    )

    if output_root is not None:
        corpus_output_dir = output_root / "corpus"
        write_graph_artifact(dedupe_result.graph, corpus_output_dir)
        write_graph_embedding_artifact(dedupe_result.graph, llm, corpus_output_dir)
        write_relation_type_proposals_artifact(
            build_relation_type_proposals(dedupe_result.graph, llm),
            corpus_output_dir,
        )
        if dedupe_result.merge_decisions:
            write_graph_merge_decisions_artifact(dedupe_result, corpus_output_dir)

    return dedupe_result.graph


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stored_document_hash(output_dir: Path, document_id: str) -> str | None:
    graph_path = output_dir / "documents" / document_id / "graph.json"
    if not graph_path.exists():
        return None
    return read_graph_artifact(graph_path).content_hash
