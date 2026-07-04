import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

from nextsearch.ingestion.artifacts import write_graph_artifact
from nextsearch.ingestion.graph.dedupe import dedupe_knowledge_graph_incremental
from nextsearch.ingestion.graph.llm_extractor import normalize_edge_id, normalize_node_id
from nextsearch.ingestion.graph.merge import (
    empty_corpus_graph,
    merge_knowledge_graphs,
    remove_document_from_graph,
)
from nextsearch.ingestion.graph.models import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    SectionGraphExtraction,
    SourceRef,
)
from nextsearch.ingestion.pipeline import ingest_pdf_to_corpus_graph
from nextsearch.llm.types import EmbeddingResponse, LLMMessage, LLMResponse
from tests.pdf_fixture import build_text_pdf


class FakeIngestLLM:
    def __init__(self) -> None:
        self.text_calls = 0
        self.graph_calls = 0
        self.embed_calls = 0

    def generate_text(
        self,
        *,
        role: str,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        self.text_calls += 1
        return LLMResponse(
            text="<!-- page: 1 -->\n# Overview\nVendor A stores data in Singapore.",
            provider="fake",
            model="fake-model",
        )

    def generate_json(
        self,
        *,
        role: str,
        messages: list[LLMMessage],
        response_model: type[Any],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> Any:
        self.graph_calls += 1
        return SectionGraphExtraction(nodes=[], edges=[])

    def embed(
        self,
        *,
        role: str,
        texts: list[str],
    ) -> EmbeddingResponse:
        self.embed_calls += 1
        return EmbeddingResponse(
            embeddings=[[1.0] for _text in texts],
            provider="fake",
            model="fake-embedding",
        )

    def embedding_provider_name(self) -> str:
        return "fake"

    def embedding_model(self) -> str:
        return "fake-embedding"


class FailingLLM:
    def generate_text(self, **kwargs: Any) -> LLMResponse:
        raise AssertionError("unchanged document should not be re-extracted")

    def generate_json(self, **kwargs: Any) -> Any:
        raise AssertionError("unchanged document should not be re-deduped")


class FakeDedupeLLM:
    def __init__(
        self,
        outputs: list[dict[str, Any]],
        *,
        embeddings: list[list[list[float]]] | None = None,
    ) -> None:
        self.outputs = list(outputs)
        self.embeddings = list(embeddings or [])
        self.calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []

    def generate_json(
        self,
        *,
        role: str,
        messages: list[LLMMessage],
        response_model: type[Any],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> Any:
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "response_model": response_model,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        return response_model.model_validate(self.outputs.pop(0))

    def embed(
        self,
        *,
        role: str,
        texts: list[str],
    ) -> EmbeddingResponse:
        self.embed_calls.append({"role": role, "texts": texts})
        if self.embeddings:
            embeddings = self.embeddings.pop(0)
        else:
            embeddings = [
                [
                    1.0 if column == row else 0.0
                    for column in range(len(texts))
                ]
                for row in range(len(texts))
            ]
        return EmbeddingResponse(
            embeddings=embeddings,
            provider="fake",
            model="fake-embedding",
        )


class KnowledgeGraphMergeTests(unittest.TestCase):
    def test_exact_node_merge_combines_document_source_refs(self) -> None:
        vendor_doc_1 = _node("doc-1", "organization", "Vendor A", "Vendor A hosts Atlas.")
        vendor_doc_2 = _node("doc-2", "organization", "Vendor A", "Vendor A stores data.")
        existing = _graph(nodes=[vendor_doc_1], edges=[])
        incoming = _graph(document_id="doc-2", nodes=[vendor_doc_2], edges=[])

        merged = merge_knowledge_graphs(existing, incoming)

        self.assertEqual(len(merged.nodes), 1)
        self.assertEqual(
            [ref.document_id for ref in merged.nodes[0].source_refs],
            ["doc-1", "doc-2"],
        )

    def test_exact_edge_merge_keeps_max_confidence_and_refs(self) -> None:
        project = _node("doc-1", "project", "Project Atlas", "Atlas uses Vendor A.")
        vendor = _node("doc-1", "organization", "Vendor A", "Vendor A hosts Atlas.")
        old_edge = _edge(project.id, "depends_on", vendor.id, "Project Atlas uses Vendor A.", 0.4, "doc-1")
        new_edge = _edge(project.id, "depends_on", vendor.id, "Project Atlas depends on Vendor A.", 0.9, "doc-2")
        existing = _graph(nodes=[project, vendor], edges=[old_edge])
        incoming = _graph(document_id="doc-2", nodes=[], edges=[new_edge])

        merged = merge_knowledge_graphs(existing, incoming)

        self.assertEqual(len(merged.edges), 1)
        self.assertEqual(merged.edges[0].confidence, 0.9)
        self.assertEqual(
            [ref.document_id for ref in merged.edges[0].source_refs],
            ["doc-1", "doc-2"],
        )

    def test_document_replacement_removes_old_refs_and_orphans(self) -> None:
        shared = _node("doc-1", "organization", "Vendor A", "Vendor A hosts Atlas.")
        shared = shared.model_copy(
            update={
                "source_refs": [
                    *_node("doc-1", "organization", "Vendor A", "Vendor A hosts Atlas.").source_refs,
                    *_node("doc-2", "organization", "Vendor A", "Vendor A stores data.").source_refs,
                ],
            }
        )
        old_only = _node("doc-1", "risk", "Legacy Risk", "Legacy Risk is retired.")
        doc_2_only = _node("doc-2", "location", "Singapore", "Singapore is the region.")
        doc_1_edge = _edge(shared.id, "has_risk", old_only.id, "Vendor A has Legacy Risk.", 0.7, "doc-1")
        doc_2_edge = _edge(shared.id, "located_in", doc_2_only.id, "Vendor A is in Singapore.", 0.8, "doc-2")
        corpus = _graph(nodes=[shared, old_only, doc_2_only], edges=[doc_1_edge, doc_2_edge])

        cleaned = remove_document_from_graph(corpus, "doc-1")

        self.assertEqual({node.id for node in cleaned.nodes}, {shared.id, doc_2_only.id})
        self.assertEqual([ref.document_id for ref in cleaned.nodes[0].source_refs], ["doc-2"])
        self.assertEqual([edge.id for edge in cleaned.edges], [doc_2_edge.id])

    def test_unchanged_document_hash_skips_extraction(self) -> None:
        pdf_bytes = build_text_pdf(["Vendor A stores data in Singapore."])
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()
        corpus = empty_corpus_graph()
        document_graph = _graph(
            document_id="doc-1",
            content_hash=content_hash,
            nodes=[],
            edges=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "sample.pdf"
            pdf_path.write_bytes(pdf_bytes)
            write_graph_artifact(document_graph, root / "documents" / "doc-1")

            result = ingest_pdf_to_corpus_graph(
                pdf_path=pdf_path,
                document_id="doc-1",
                llm=FailingLLM(),  # type: ignore[arg-type]
                corpus_graph=corpus,
                output_dir=root,
            )

        self.assertIs(result, corpus)

    def test_corpus_ingestion_writes_document_and_corpus_artifacts(self) -> None:
        llm = FakeIngestLLM()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "sample.pdf"
            pdf_path.write_bytes(build_text_pdf(["Vendor A stores data in Singapore."]))

            ingest_pdf_to_corpus_graph(
                pdf_path=pdf_path,
                document_id="doc-1",
                llm=llm,  # type: ignore[arg-type]
                corpus_graph=None,
                output_dir=root,
            )

            self.assertTrue((root / "documents" / "doc-1" / "document.md").exists())
            self.assertTrue((root / "documents" / "doc-1" / "manifest.json").exists())
            self.assertTrue((root / "documents" / "doc-1" / "graph.json").exists())
            self.assertTrue((root / "documents" / "doc-1" / "graph_embeddings.json").exists())
            self.assertTrue((root / "corpus" / "graph.json").exists())
            self.assertTrue((root / "corpus" / "graph_embeddings.json").exists())

        self.assertEqual(llm.text_calls, 1)
        self.assertEqual(llm.graph_calls, 1)
        self.assertEqual(llm.embed_calls, 0)

    def test_incremental_dedupe_only_compares_incoming_candidates(self) -> None:
        vendor = _node("doc-old", "organization", "Vendor A", "Vendor A is mentioned.")
        vendor_ltd = _node("doc-old", "organization", "Vendor A Ltd.", "Vendor A Ltd. is mentioned.")
        supplier = _node("doc-old", "organization", "Supplier B", "Supplier B is mentioned.")
        supplier_ltd = _node("doc-new", "organization", "Supplier B Ltd.", "Supplier B Ltd. is mentioned.")
        graph = _graph(nodes=[vendor, vendor_ltd, supplier, supplier_ltd], edges=[])
        llm = FakeDedupeLLM(
            [{"decision": "uncertain", "confidence": 0.9, "reason": "Needs more evidence."}]
        )

        dedupe_knowledge_graph_incremental(
            graph,
            llm,  # type: ignore[arg-type]
            incoming_node_ids={supplier_ltd.id},
        )

        self.assertEqual(len(llm.calls), 1)
        prompt = llm.calls[0]["messages"][1].content
        self.assertIn("Supplier B", prompt)
        self.assertNotIn("Vendor A Ltd.", prompt)


def _graph(
    *,
    document_id: str = "corpus",
    content_hash: str = "",
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> KnowledgeGraph:
    return KnowledgeGraph(
        document_id=document_id,
        content_hash=content_hash,
        source_path=document_id,
        page_count=1,
        nodes=nodes,
        edges=edges,
    )


def _node(document_id: str, node_type: str, name: str, quote: str) -> GraphNode:
    return GraphNode(
        id=normalize_node_id(node_type, name),
        type=node_type,  # type: ignore[arg-type]
        name=name,
        source_refs=[_source_ref(document_id, quote)],
    )


def _edge(
    source_node_id: str,
    relation_type: str,
    target_node_id: str,
    quote: str,
    confidence: float,
    document_id: str,
) -> GraphEdge:
    return GraphEdge(
        id=normalize_edge_id(source_node_id, relation_type, target_node_id),
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        relation_type=relation_type,  # type: ignore[arg-type]
        raw_relation=relation_type,
        confidence=confidence,
        source_refs=[_source_ref(document_id, quote)],
    )


def _source_ref(document_id: str, quote: str) -> SourceRef:
    return SourceRef(
        document_id=document_id,
        section_id="section-0001",
        heading="Overview",
        quote=quote,
    )


if __name__ == "__main__":
    unittest.main()
