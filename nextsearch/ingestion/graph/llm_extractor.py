from __future__ import annotations

import hashlib
import re

from nextsearch.ingestion.errors import GraphExtractionError
from nextsearch.ingestion.graph.models import (
    EntityType,
    ExtractedEdge,
    ExtractedNode,
    ExtractedSourceRef,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    NodeRef,
    SectionGraphExtraction,
    SourceRef,
)
from nextsearch.ingestion.graph.prompts import build_graph_extraction_messages
from nextsearch.ingestion.markdown.sections import split_markdown_into_sections
from nextsearch.ingestion.models import DocumentSection, MarkdownDocument
from nextsearch.llm.service import LLMService


_RISKY_NODE_ID_SYMBOL_RE = re.compile(r"[+#/\\&@%]")
_TRAILING_SENTENCE_PUNCTUATION = ".,;:!?"


def extract_knowledge_graph_from_markdown(
    document: MarkdownDocument,
    llm: LLMService,
    *,
    document_id: str,
    content_hash: str,
    role: str = "graph_extraction",
) -> KnowledgeGraph:
    sections = split_markdown_into_sections(document)
    nodes: dict[str, GraphNode] = {}
    edges: dict[str, GraphEdge] = {}

    for section in sections:
        if not section.text.strip():
            continue

        try:
            extraction = llm.generate_json(
                role=role,
                messages=build_graph_extraction_messages(section),
                response_model=SectionGraphExtraction,
                temperature=0,
            )
        except Exception as exc:
            raise GraphExtractionError(
                f"Graph extraction failed for section {section.id}"
            ) from exc

        _merge_extraction(
            nodes=nodes,
            edges=edges,
            extraction=extraction,
            section=section,
            document_id=document_id,
        )

    return KnowledgeGraph(
        document_id=document_id,
        content_hash=content_hash,
        source_path=str(document.source_path),
        page_count=document.page_count,
        nodes=list(nodes.values()),
        edges=list(edges.values()),
    )


def _merge_extraction(
    *,
    nodes: dict[str, GraphNode],
    edges: dict[str, GraphEdge],
    extraction: SectionGraphExtraction,
    section: DocumentSection,
    document_id: str,
) -> None:
    for extracted_node in extraction.nodes:
        _merge_node(
            nodes,
            node_type=extracted_node.type,
            name=extracted_node.name,
            description=extracted_node.description,
            source_refs=_normalize_source_refs(
                extracted_node.source_refs,
                section,
                document_id=document_id,
            ),
        )

    for extracted_edge in extraction.edges:
        edge_source_refs = _normalize_source_refs(
            extracted_edge.source_refs,
            section,
            document_id=document_id,
        )
        source_node_id = _merge_node_ref(nodes, extracted_edge.source, edge_source_refs)
        target_node_id = _merge_node_ref(nodes, extracted_edge.target, edge_source_refs)
        _merge_edge(
            edges,
            extracted_edge=extracted_edge,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            source_refs=edge_source_refs,
        )


def _merge_node_ref(
    nodes: dict[str, GraphNode],
    node_ref: NodeRef,
    source_refs: list[SourceRef],
) -> str:
    return _merge_node(
        nodes,
        node_type=node_ref.type,
        name=node_ref.name,
        description=None,
        source_refs=source_refs,
    )


def _merge_node(
    nodes: dict[str, GraphNode],
    *,
    node_type: EntityType,
    name: str,
    description: str | None,
    source_refs: list[SourceRef],
) -> str:
    node_id = normalize_node_id(node_type, name)
    existing = nodes.get(node_id)
    if existing is None:
        nodes[node_id] = GraphNode(
            id=node_id,
            type=node_type,
            name=name.strip(),
            description=_clean_optional_text(description),
            source_refs=source_refs,
        )
        return node_id

    nodes[node_id] = existing.model_copy(
        update={
            "description": existing.description or _clean_optional_text(description),
            "source_refs": _merge_source_refs(existing.source_refs, source_refs),
        }
    )
    return node_id


def _merge_edge(
    edges: dict[str, GraphEdge],
    *,
    extracted_edge: ExtractedEdge,
    source_node_id: str,
    target_node_id: str,
    source_refs: list[SourceRef],
) -> None:
    edge_id = normalize_edge_id(
        source_node_id,
        extracted_edge.relation_type,
        target_node_id,
    )
    existing = edges.get(edge_id)
    if existing is None:
        edges[edge_id] = GraphEdge(
            id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relation_type=extracted_edge.relation_type,
            raw_relation=extracted_edge.raw_relation.strip(),
            description=_clean_optional_text(extracted_edge.description),
            confidence=extracted_edge.confidence,
            source_refs=source_refs,
        )
        return

    edges[edge_id] = existing.model_copy(
        update={
            "description": existing.description
            or _clean_optional_text(extracted_edge.description),
            "confidence": max(existing.confidence, extracted_edge.confidence),
            "source_refs": _merge_source_refs(existing.source_refs, source_refs),
        }
    )


def _normalize_source_refs(
    source_refs: list[ExtractedSourceRef],
    section: DocumentSection,
    *,
    document_id: str,
) -> list[SourceRef]:
    normalized: list[SourceRef] = []
    for source_ref in source_refs:
        quote = source_ref.quote.strip()
        if quote:
            normalized.append(
                SourceRef(
                    document_id=document_id,
                    section_id=section.id,
                    heading=section.heading,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    quote=quote,
                )
            )
    return normalized


def _merge_source_refs(
    existing: list[SourceRef],
    incoming: list[SourceRef],
) -> list[SourceRef]:
    merged = list(existing)
    seen = {
        (
            source_ref.document_id,
            source_ref.section_id,
            source_ref.heading,
            source_ref.page_start,
            source_ref.page_end,
            source_ref.quote,
        )
        for source_ref in existing
    }
    for source_ref in incoming:
        key = (
            source_ref.document_id,
            source_ref.section_id,
            source_ref.heading,
            source_ref.page_start,
            source_ref.page_end,
            source_ref.quote,
        )
        if key not in seen:
            merged.append(source_ref)
            seen.add(key)
    return merged


def normalize_node_id(node_type: str, name: str) -> str:
    slug = _slugify(name)
    if _has_risky_node_id_symbol(name):
        return f"{node_type}:{slug}-{_node_id_hash(node_type, name)}"
    return f"{node_type}:{slug}"


def normalize_edge_id(
    source_node_id: str,
    relation_type: str,
    target_node_id: str,
) -> str:
    return f"{source_node_id}|{relation_type}|{target_node_id}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def _has_risky_node_id_symbol(value: str) -> bool:
    return _RISKY_NODE_ID_SYMBOL_RE.search(value) is not None


def _node_id_hash(node_type: str, name: str) -> str:
    canonical_name = _canonicalize_node_name_for_id_hash(name)
    payload = f"{node_type}:{canonical_name}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _canonicalize_node_name_for_id_hash(name: str) -> str:
    canonical = " ".join(name.casefold().strip().split())
    canonical = canonical.rstrip(_TRAILING_SENTENCE_PUNCTUATION).strip()
    return canonical


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
