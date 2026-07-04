import json
import tempfile
import unittest
from pathlib import Path

from nextsearch.ingestion.artifacts import write_graph_artifact
from nextsearch.ingestion.graph.models import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    SourceRef,
)
from nextsearch.retrieval import (
    JsonGraphStore,
    MarkdownArtifactSourceStore,
    build_evidence_records,
    search_graph,
)


class RetrievalTests(unittest.TestCase):
    def test_graph_store_loads_graph_artifact(self) -> None:
        graph = _graph()

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_dir = Path(tmpdir) / "corpus"
            write_graph_artifact(graph, graph_dir)

            loaded = JsonGraphStore.from_artifact_dir(graph_dir).load_graph()

        self.assertEqual(loaded.document_id, "corpus")
        self.assertEqual(
            [node.name for node in loaded.nodes],
            ["Project Atlas", "Vendor A", "Data residency issue"],
        )

    def test_graph_search_matches_nodes_and_expands_related_edges(self) -> None:
        result = search_graph(
            _graph(),
            search_terms=["Atlas"],
            relation_types=["has_risk"],
        )

        self.assertIn("project:project-atlas", {node.id for node in result.nodes})
        self.assertIn("risk:data-residency-issue", {node.id for node in result.nodes})
        self.assertEqual([edge.relation_type for edge in result.edges], ["has_risk"])

    def test_evidence_builder_maps_source_refs_to_markdown_sections(self) -> None:
        graph = _graph()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_document_artifacts(root)
            result = search_graph(
                graph,
                search_terms=["Atlas"],
                relation_types=["has_risk"],
            )

            evidence = build_evidence_records(
                result,
                MarkdownArtifactSourceStore(root),
            )

        self.assertTrue(evidence)
        self.assertEqual(evidence[0].citation.document_id, "doc-1")
        self.assertEqual(evidence[0].citation.section_id, "section-0001")
        self.assertIn("Data residency issue", evidence[0].snippet)


def _write_document_artifacts(root: Path) -> None:
    document_dir = root / "documents" / "doc-1"
    document_dir.mkdir(parents=True)
    (document_dir / "document.md").write_text(
        "<!-- page: 1 -->\n"
        "# Overview\n"
        "Project Atlas depends on Vendor A.\n"
        "Project Atlas has Data residency issue.\n",
        encoding="utf-8",
    )
    (document_dir / "manifest.json").write_text(
        json.dumps({"source_path": "sample.pdf", "page_count": 1, "batches": []}),
        encoding="utf-8",
    )


def _graph() -> KnowledgeGraph:
    project = GraphNode(
        id="project:project-atlas",
        type="project",
        name="Project Atlas",
        aliases=[],
        source_refs=[_source_ref("Project Atlas depends on Vendor A.")],
    )
    vendor = GraphNode(
        id="organization:vendor-a",
        type="organization",
        name="Vendor A",
        aliases=[],
        source_refs=[_source_ref("Project Atlas depends on Vendor A.")],
    )
    risk = GraphNode(
        id="risk:data-residency-issue",
        type="risk",
        name="Data residency issue",
        aliases=[],
        source_refs=[_source_ref("Project Atlas has Data residency issue.")],
    )
    return KnowledgeGraph(
        document_id="corpus",
        content_hash="",
        source_path="",
        page_count=1,
        nodes=[project, vendor, risk],
        edges=[
            GraphEdge(
                id="project:project-atlas|depends_on|organization:vendor-a",
                source_node_id=project.id,
                target_node_id=vendor.id,
                relation_type="depends_on",
                raw_relation="Project Atlas depends on Vendor A.",
                confidence=0.9,
                source_refs=[_source_ref("Project Atlas depends on Vendor A.")],
            ),
            GraphEdge(
                id="project:project-atlas|has_risk|risk:data-residency-issue",
                source_node_id=project.id,
                target_node_id=risk.id,
                relation_type="has_risk",
                raw_relation="Project Atlas has Data residency issue.",
                confidence=0.8,
                source_refs=[_source_ref("Project Atlas has Data residency issue.")],
            ),
        ],
    )


def _source_ref(quote: str) -> SourceRef:
    return SourceRef(
        document_id="doc-1",
        section_id="section-0001",
        heading="Overview",
        page_start=1,
        page_end=1,
        quote=quote,
    )


if __name__ == "__main__":
    unittest.main()
