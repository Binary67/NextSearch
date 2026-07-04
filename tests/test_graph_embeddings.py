import unittest

from nextsearch.ingestion.graph.embeddings import build_graph_embedding_index
from nextsearch.ingestion.graph.llm_extractor import normalize_edge_id, normalize_node_id
from nextsearch.ingestion.graph.models import GraphEdge, GraphNode, KnowledgeGraph, SourceRef
from nextsearch.llm.types import EmbeddingResponse


class FakeEmbeddingLLM:
    def __init__(self) -> None:
        self.embed_calls: list[dict[str, object]] = []
        self.next_value = 1.0

    def embedding_provider_name(self) -> str:
        return "fake"

    def embedding_model(self) -> str:
        return "fake-embedding"

    def embed(self, *, role: str, texts: list[str]) -> EmbeddingResponse:
        self.embed_calls.append({"role": role, "texts": texts})
        embeddings: list[list[float]] = []
        for _text in texts:
            embeddings.append([self.next_value, 0.0])
            self.next_value += 1.0
        return EmbeddingResponse(
            embeddings=embeddings,
            provider="fake",
            model="fake-embedding",
        )


class GraphEmbeddingTests(unittest.TestCase):
    def test_build_graph_embedding_index_embeds_nodes_and_edges(self) -> None:
        llm = FakeEmbeddingLLM()
        graph = _graph(node_description="Atlas is a project.")

        index = build_graph_embedding_index(graph, llm)  # type: ignore[arg-type]

        self.assertEqual(index.graph_document_id, "doc-1")
        self.assertEqual(index.provider, "fake")
        self.assertEqual(index.model, "fake-embedding")
        self.assertEqual(
            [(item.item_type, item.item_id) for item in index.items],
            [
                ("node", "project:project-atlas"),
                ("node", "organization:vendor-a"),
                ("edge", "project:project-atlas|depends_on|organization:vendor-a"),
            ],
        )
        self.assertEqual(len(llm.embed_calls), 1)
        self.assertEqual(llm.embed_calls[0]["role"], "graph_embedding")
        self.assertEqual(len(llm.embed_calls[0]["texts"]), 3)

    def test_build_graph_embedding_index_reuses_unchanged_items(self) -> None:
        first_llm = FakeEmbeddingLLM()
        graph = _graph(node_description="Atlas is a project.")
        first_index = build_graph_embedding_index(graph, first_llm)  # type: ignore[arg-type]
        second_llm = FakeEmbeddingLLM()

        second_index = build_graph_embedding_index(
            _graph(node_description="Atlas stores regulated records."),
            second_llm,  # type: ignore[arg-type]
            previous_index=first_index,
        )

        self.assertEqual(len(second_llm.embed_calls), 1)
        self.assertEqual(len(second_llm.embed_calls[0]["texts"]), 1)
        changed_item = second_index.items[0]
        unchanged_items = second_index.items[1:]
        self.assertNotEqual(changed_item.text_hash, first_index.items[0].text_hash)
        self.assertEqual(changed_item.embedding, [1.0, 0.0])
        self.assertEqual(
            [item.embedding for item in unchanged_items],
            [item.embedding for item in first_index.items[1:]],
        )


def _graph(*, node_description: str) -> KnowledgeGraph:
    project = GraphNode(
        id=normalize_node_id("project", "Project Atlas"),
        type="project",
        name="Project Atlas",
        description=node_description,
        source_refs=[_source_ref("Project Atlas depends on Vendor A.")],
    )
    vendor = GraphNode(
        id=normalize_node_id("organization", "Vendor A"),
        type="organization",
        name="Vendor A",
        source_refs=[_source_ref("Vendor A hosts Project Atlas.")],
    )
    edge = GraphEdge(
        id=normalize_edge_id(project.id, "depends_on", vendor.id),
        source_node_id=project.id,
        target_node_id=vendor.id,
        relation_type="depends_on",
        raw_relation="Project Atlas depends on Vendor A.",
        confidence=0.9,
        source_refs=[_source_ref("Project Atlas depends on Vendor A.")],
    )
    return KnowledgeGraph(
        document_id="doc-1",
        content_hash="hash-1",
        source_path="sample.pdf",
        page_count=1,
        nodes=[project, vendor],
        edges=[edge],
    )


def _source_ref(quote: str) -> SourceRef:
    return SourceRef(
        document_id="doc-1",
        section_id="section-0001",
        heading="Overview",
        quote=quote,
    )


if __name__ == "__main__":
    unittest.main()
