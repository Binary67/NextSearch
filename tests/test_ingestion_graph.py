import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from nextsearch.ingestion.graph.llm_extractor import (
    extract_knowledge_graph_from_markdown,
)
from nextsearch.ingestion.graph.models import (
    ExtractedEdge,
    ExtractedNode,
    NodeRef,
    SectionGraphExtraction,
    SourceRef,
)
from nextsearch.ingestion.models import MarkdownDocument
from nextsearch.ingestion.pipeline import extract_pdf_to_knowledge_graph
from nextsearch.llm.types import LLMMessage, LLMResponse
from tests.pdf_fixture import build_text_pdf


class FakeGraphLLM:
    def __init__(self, outputs: list[SectionGraphExtraction]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

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
        return self.outputs.pop(0)


class FakePDFGraphLLM:
    def __init__(self) -> None:
        self.graph_calls = 0

    def generate_text(
        self,
        *,
        role: str,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        text = "\n".join(
            f"<!-- page: {page_number} -->\n# Page {page_number}\nVendor A stores data in Singapore."
            for page_number in _page_numbers(messages[-1].content)
        )
        return LLMResponse(text=text, provider="fake", model="fake-model")

    def generate_json(
        self,
        *,
        role: str,
        messages: list[LLMMessage],
        response_model: type[Any],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> SectionGraphExtraction:
        self.graph_calls += 1
        return SectionGraphExtraction(nodes=[], edges=[])


class KnowledgeGraphExtractionTests(unittest.TestCase):
    def test_extract_graph_uses_graph_role_and_dedupes_identical_edges(self) -> None:
        document = _document(
            "<!-- page: 1 -->\n"
            "# Overview\n"
            "Project Atlas depends on Vendor A.\n\n"
            "<!-- page: 2 -->\n"
            "## Dependencies\n"
            "Project Atlas depends on Vendor A for hosting.\n"
        )
        llm = FakeGraphLLM(
            [
                SectionGraphExtraction(
                    nodes=[
                        _node("project", "Project Atlas", "Project Atlas depends on Vendor A."),
                        _node("organization", "Vendor A", "Project Atlas depends on Vendor A."),
                    ],
                    edges=[
                        _edge(
                            source_type="project",
                            source_name="Project Atlas",
                            relation_type="depends_on",
                            target_type="organization",
                            target_name="Vendor A",
                            quote="Project Atlas depends on Vendor A.",
                            confidence=0.7,
                        )
                    ],
                ),
                SectionGraphExtraction(
                    nodes=[
                        _node("project", "Project Atlas", "Project Atlas depends on Vendor A for hosting."),
                        _node("organization", "Vendor A", "Project Atlas depends on Vendor A for hosting."),
                    ],
                    edges=[
                        _edge(
                            source_type="project",
                            source_name="Project Atlas",
                            relation_type="depends_on",
                            target_type="organization",
                            target_name="Vendor A",
                            quote="Project Atlas depends on Vendor A for hosting.",
                            confidence=0.9,
                        )
                    ],
                ),
            ]
        )

        graph = extract_knowledge_graph_from_markdown(document, llm)  # type: ignore[arg-type]

        self.assertEqual([call["role"] for call in llm.calls], ["graph_extraction", "graph_extraction"])
        self.assertIs(llm.calls[0]["response_model"], SectionGraphExtraction)
        self.assertEqual(llm.calls[0]["temperature"], 0)
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)
        edge = graph.edges[0]
        self.assertEqual(edge.id, "project:project-atlas|depends_on|organization:vendor-a")
        self.assertEqual(edge.confidence, 0.9)
        self.assertEqual(len(edge.source_refs), 2)
        self.assertEqual([ref.section_id for ref in edge.source_refs], ["section-0001", "section-0002"])

    def test_extract_graph_does_not_merge_similar_node_names(self) -> None:
        document = _document(
            "<!-- page: 1 -->\n"
            "# Vendors\n"
            "Vendor A and Vendor A Ltd. are mentioned separately.\n"
        )
        llm = FakeGraphLLM(
            [
                SectionGraphExtraction(
                    nodes=[
                        _node("organization", "Vendor A", "Vendor A is mentioned."),
                        _node("organization", "Vendor A Ltd.", "Vendor A Ltd. is mentioned."),
                        _node("location", "Singapore", "Singapore is mentioned."),
                    ],
                    edges=[
                        _edge(
                            source_type="organization",
                            source_name="Vendor A",
                            relation_type="located_in",
                            target_type="location",
                            target_name="Singapore",
                            quote="Vendor A is located in Singapore.",
                        ),
                        _edge(
                            source_type="organization",
                            source_name="Vendor A Ltd.",
                            relation_type="located_in",
                            target_type="location",
                            target_name="Singapore",
                            quote="Vendor A Ltd. is located in Singapore.",
                        ),
                    ],
                )
            ]
        )

        graph = extract_knowledge_graph_from_markdown(document, llm)  # type: ignore[arg-type]

        self.assertIn("organization:vendor-a", {node.id for node in graph.nodes})
        self.assertIn("organization:vendor-a-ltd", {node.id for node in graph.nodes})
        self.assertEqual(len(graph.edges), 2)

    def test_extract_pdf_to_knowledge_graph_writes_graph_artifact(self) -> None:
        llm = FakePDFGraphLLM()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "sample.pdf"
            output_dir = root / "out"
            pdf_path.write_bytes(build_text_pdf(["Vendor A stores data in Singapore."]))

            graph = extract_pdf_to_knowledge_graph(
                pdf_path,
                llm,  # type: ignore[arg-type]
                output_dir=output_dir,
            )

            graph_path = output_dir / "graph.json"
            payload = json.loads(graph_path.read_text(encoding="utf-8"))

            self.assertEqual(graph.schema_version, 1)
            self.assertTrue(graph_path.exists())
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["source_path"], str(pdf_path))
            self.assertIn("nodes", payload)
            self.assertIn("edges", payload)
            self.assertEqual(llm.graph_calls, 1)


def _document(markdown: str) -> MarkdownDocument:
    return MarkdownDocument(
        markdown=markdown,
        source_path=Path("sample.pdf"),
        page_count=2,
        batches=(),
    )


def _node(node_type: str, name: str, quote: str) -> ExtractedNode:
    return ExtractedNode(
        type=node_type,  # type: ignore[arg-type]
        name=name,
        source_refs=[_source_ref(quote)],
    )


def _edge(
    *,
    source_type: str,
    source_name: str,
    relation_type: str,
    target_type: str,
    target_name: str,
    quote: str,
    confidence: float = 0.8,
) -> ExtractedEdge:
    return ExtractedEdge(
        source=NodeRef(type=source_type, name=source_name),  # type: ignore[arg-type]
        target=NodeRef(type=target_type, name=target_name),  # type: ignore[arg-type]
        relation_type=relation_type,  # type: ignore[arg-type]
        raw_relation=relation_type,
        confidence=confidence,
        source_refs=[_source_ref(quote)],
    )


def _source_ref(quote: str) -> SourceRef:
    return SourceRef(
        section_id="ignored",
        heading="ignored",
        quote=quote,
    )


def _page_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for line in text.splitlines():
        if line.startswith("<!-- page: ") and line.endswith(" -->"):
            numbers.append(int(line.removeprefix("<!-- page: ").removesuffix(" -->")))
    return numbers


if __name__ == "__main__":
    unittest.main()
