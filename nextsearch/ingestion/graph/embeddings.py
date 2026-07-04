from __future__ import annotations

import hashlib
import json
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from nextsearch.ingestion.graph.models import GraphEdge, GraphNode, KnowledgeGraph, SourceRef
from nextsearch.llm.service import LLMService


GRAPH_EMBEDDING_SCHEMA_VERSION = 1
GRAPH_EMBEDDING_TEXT_VERSION = 1
MAX_SOURCE_QUOTES = 4
MAX_QUOTE_CHARS = 260


class GraphEmbeddingIndexError(Exception):
    """Raised when a graph embedding artifact cannot serve the current graph."""


class GraphEmbeddingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_type: Literal["node", "edge"]
    item_id: str
    text_hash: str
    embedding: list[float] = Field(min_length=1)


class GraphEmbeddingIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = GRAPH_EMBEDDING_SCHEMA_VERSION
    embedding_text_version: int = GRAPH_EMBEDDING_TEXT_VERSION
    graph_document_id: str
    graph_fingerprint: str
    provider: str
    model: str
    items: list[GraphEmbeddingItem] = Field(default_factory=list)


class GraphEmbeddingInput(NamedTuple):
    item_type: Literal["node", "edge"]
    item_id: str
    text: str

    @property
    def text_hash(self) -> str:
        return _sha256_text(self.text)


def build_graph_embedding_index(
    graph: KnowledgeGraph,
    llm: LLMService,
    *,
    previous_index: GraphEmbeddingIndex | None = None,
) -> GraphEmbeddingIndex:
    provider = llm.embedding_provider_name()
    model = llm.embedding_model()
    inputs = graph_embedding_inputs(graph)
    reusable_items = _reusable_items(previous_index, provider=provider, model=model)

    items_by_key: dict[tuple[str, str, str], GraphEmbeddingItem] = {}
    pending_inputs: list[GraphEmbeddingInput] = []
    for item_input in inputs:
        key = (item_input.item_type, item_input.item_id, item_input.text_hash)
        reusable_item = reusable_items.get(key)
        if reusable_item is None:
            pending_inputs.append(item_input)
            continue
        items_by_key[key] = reusable_item

    if pending_inputs:
        response = llm.embed(
            role="graph_embedding",
            texts=[item_input.text for item_input in pending_inputs],
        )
        if len(response.embeddings) != len(pending_inputs):
            raise GraphEmbeddingIndexError(
                "Graph embedding generation returned wrong embedding count"
            )
        for item_input, embedding in zip(pending_inputs, response.embeddings, strict=True):
            key = (item_input.item_type, item_input.item_id, item_input.text_hash)
            items_by_key[key] = GraphEmbeddingItem(
                item_type=item_input.item_type,
                item_id=item_input.item_id,
                text_hash=item_input.text_hash,
                embedding=embedding,
            )

    return GraphEmbeddingIndex(
        graph_document_id=graph.document_id,
        graph_fingerprint=graph_fingerprint(graph),
        provider=provider,
        model=model,
        items=[
            items_by_key[(item_input.item_type, item_input.item_id, item_input.text_hash)]
            for item_input in inputs
        ],
    )


def validate_graph_embedding_index(
    graph: KnowledgeGraph,
    index: GraphEmbeddingIndex,
    *,
    provider: str,
    model: str,
) -> dict[tuple[str, str], GraphEmbeddingItem]:
    if index.schema_version != GRAPH_EMBEDDING_SCHEMA_VERSION:
        raise GraphEmbeddingIndexError(
            f"Graph embedding index schema version {index.schema_version} is not supported"
        )
    if index.embedding_text_version != GRAPH_EMBEDDING_TEXT_VERSION:
        raise GraphEmbeddingIndexError(
            "Graph embedding index text version is stale"
        )
    if index.graph_document_id != graph.document_id:
        raise GraphEmbeddingIndexError(
            "Graph embedding index document id does not match graph"
        )
    if index.graph_fingerprint != graph_fingerprint(graph):
        raise GraphEmbeddingIndexError("Graph embedding index is stale for this graph")
    if index.provider != provider:
        raise GraphEmbeddingIndexError(
            "Graph embedding index provider does not match configured provider"
        )
    if index.model != model:
        raise GraphEmbeddingIndexError(
            "Graph embedding index model does not match configured embedding model"
        )

    item_by_key: dict[tuple[str, str], GraphEmbeddingItem] = {}
    for item in index.items:
        key = (item.item_type, item.item_id)
        if key in item_by_key:
            raise GraphEmbeddingIndexError(
                f"Graph embedding index has duplicate item {item.item_type}:{item.item_id}"
            )
        item_by_key[key] = item

    for item_input in graph_embedding_inputs(graph):
        key = (item_input.item_type, item_input.item_id)
        item = item_by_key.get(key)
        if item is None:
            raise GraphEmbeddingIndexError(
                f"Graph embedding index is missing {item_input.item_type} {item_input.item_id}"
            )
        if item.text_hash != item_input.text_hash:
            raise GraphEmbeddingIndexError(
                "Graph embedding index text is stale for "
                f"{item_input.item_type} {item_input.item_id}"
            )

    return item_by_key


def graph_embedding_inputs(graph: KnowledgeGraph) -> list[GraphEmbeddingInput]:
    node_by_id = {node.id: node for node in graph.nodes}
    inputs = [
        GraphEmbeddingInput("node", node.id, node_embedding_text(node))
        for node in graph.nodes
    ]
    inputs.extend(
        GraphEmbeddingInput("edge", edge.id, edge_embedding_text(edge, node_by_id))
        for edge in graph.edges
    )
    return inputs


def graph_fingerprint(graph: KnowledgeGraph) -> str:
    payload = json.dumps(
        graph.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(payload)


def node_embedding_text(node: GraphNode) -> str:
    aliases = ", ".join(node.aliases) if node.aliases else "none"
    description = node.description or "none"
    quotes = "\n".join(_source_quote_lines(node.source_refs)) or "- none"
    return (
        "item: node\n"
        f"id: {node.id}\n"
        f"type: {node.type}\n"
        f"name: {node.name}\n"
        f"aliases: {aliases}\n"
        f"description: {description}\n"
        f"source quotes:\n{quotes}"
    )


def edge_embedding_text(edge: GraphEdge, node_by_id: dict[str, GraphNode]) -> str:
    source = node_by_id[edge.source_node_id]
    target = node_by_id[edge.target_node_id]
    description = edge.description or "none"
    quotes = "\n".join(_source_quote_lines(edge.source_refs)) or "- none"
    return (
        "item: edge\n"
        f"id: {edge.id}\n"
        f"source: {source.name} ({source.type})\n"
        f"relation_type: {edge.relation_type}\n"
        f"target: {target.name} ({target.type})\n"
        f"raw_relation: {edge.raw_relation}\n"
        f"description: {description}\n"
        f"source quotes:\n{quotes}"
    )


def _reusable_items(
    previous_index: GraphEmbeddingIndex | None,
    *,
    provider: str,
    model: str,
) -> dict[tuple[str, str, str], GraphEmbeddingItem]:
    if (
        previous_index is None
        or previous_index.schema_version != GRAPH_EMBEDDING_SCHEMA_VERSION
        or previous_index.embedding_text_version != GRAPH_EMBEDDING_TEXT_VERSION
        or previous_index.provider != provider
        or previous_index.model != model
    ):
        return {}

    return {
        (item.item_type, item.item_id, item.text_hash): item
        for item in previous_index.items
    }


def _source_quote_lines(source_refs: list[SourceRef]) -> list[str]:
    lines: list[str] = []
    for source_ref in source_refs[:MAX_SOURCE_QUOTES]:
        quote = source_ref.quote.strip()
        if len(quote) > MAX_QUOTE_CHARS:
            quote = quote[:MAX_QUOTE_CHARS].rstrip() + "..."
        lines.append(f"- {quote}")
    return lines


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
