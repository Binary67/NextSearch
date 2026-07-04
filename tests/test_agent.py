import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from nextsearch.agent import build_query_agent
from nextsearch.agent.models import AnswerDraft, GraphSearchDecision, QueryPlan
from nextsearch.ingestion.graph.models import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    SourceRef,
)
from nextsearch.llm.types import LLMMessage
from nextsearch.retrieval import MarkdownArtifactSourceStore


class FakeAgentLLM:
    def __init__(
        self,
        *,
        decisions: list[GraphSearchDecision] | None = None,
        answer: AnswerDraft | None = None,
    ) -> None:
        self.decisions = list(decisions or [])
        self.answer = answer or AnswerDraft(
            answer="Project Atlas has a data residency risk.",
            cited_evidence_ids=[1],
        )
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
        if response_model is QueryPlan:
            return QueryPlan(search_terms=["Project Atlas"], relation_types=["has_risk"])
        if response_model is GraphSearchDecision:
            return self.decisions.pop(0)
        if response_model is AnswerDraft:
            return self.answer
        raise AssertionError(f"Unexpected response model {response_model}")


class CountingGraphStore:
    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph
        self.calls = 0

    def load_graph(self) -> KnowledgeGraph:
        self.calls += 1
        return self.graph


class MissingSourceStore:
    def get_section(self, *, document_id: str, section_id: str) -> None:
        return None


class AgentTests(unittest.TestCase):
    def test_agent_answers_with_source_citations(self) -> None:
        llm = FakeAgentLLM(
            decisions=[
                GraphSearchDecision(next_step="answer", reason="Enough evidence.")
            ]
        )
        graph_store = CountingGraphStore(_graph())

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_document_artifacts(root)
            agent = build_query_agent(
                llm=llm,  # type: ignore[arg-type]
                graph_store=graph_store,
                source_store=MarkdownArtifactSourceStore(root),
            )

            result = agent.invoke({"query": "What risks are related to Project Atlas?"})

        self.assertEqual(set(result), {"answer", "citations", "evidence"})
        self.assertIn("data residency", result["answer"])
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0].document_id, "doc-1")
        self.assertEqual(graph_store.calls, 1)

    def test_agent_loops_when_decision_requests_more_graph_search(self) -> None:
        llm = FakeAgentLLM(
            decisions=[
                GraphSearchDecision(
                    next_step="search_more",
                    search_terms=["Data residency issue"],
                    relation_types=["has_risk"],
                    reason="Need the specific risk.",
                )
            ]
        )
        graph_store = CountingGraphStore(_graph())

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_document_artifacts(root)
            agent = build_query_agent(
                llm=llm,  # type: ignore[arg-type]
                graph_store=graph_store,
                source_store=MarkdownArtifactSourceStore(root),
                max_search_iterations=2,
            )

            result = agent.invoke({"query": "What risks are related to Project Atlas?"})

        self.assertEqual(graph_store.calls, 2)
        self.assertTrue(result["citations"])
        self.assertEqual(
            [call["role"] for call in llm.calls],
            ["query_planning", "graph_search_decision", "answer_generation"],
        )

    def test_agent_stops_at_max_search_iterations(self) -> None:
        llm = FakeAgentLLM()
        graph_store = CountingGraphStore(_graph())

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_document_artifacts(root)
            agent = build_query_agent(
                llm=llm,  # type: ignore[arg-type]
                graph_store=graph_store,
                source_store=MarkdownArtifactSourceStore(root),
                max_search_iterations=1,
            )

            agent.invoke({"query": "What risks are related to Project Atlas?"})

        self.assertEqual(graph_store.calls, 1)
        self.assertEqual(
            [call["role"] for call in llm.calls],
            ["query_planning", "answer_generation"],
        )

    def test_agent_does_not_cite_without_source_evidence(self) -> None:
        llm = FakeAgentLLM(
            decisions=[
                GraphSearchDecision(next_step="answer", reason="No source evidence.")
            ]
        )
        agent = build_query_agent(
            llm=llm,  # type: ignore[arg-type]
            graph_store=CountingGraphStore(_graph()),
            source_store=MissingSourceStore(),  # type: ignore[arg-type]
        )

        result = agent.invoke({"query": "What risks are related to Project Atlas?"})

        self.assertEqual(result["citations"], [])
        self.assertEqual(result["evidence"], [])
        self.assertIn("not have enough source evidence", result["answer"])
        self.assertEqual(
            [call["role"] for call in llm.calls],
            ["query_planning", "graph_search_decision"],
        )


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
        source_refs=[_source_ref("Project Atlas depends on Vendor A.")],
    )
    risk = GraphNode(
        id="risk:data-residency-issue",
        type="risk",
        name="Data residency issue",
        source_refs=[_source_ref("Project Atlas has Data residency issue.")],
    )
    return KnowledgeGraph(
        document_id="corpus",
        content_hash="",
        source_path="",
        page_count=1,
        nodes=[project, risk],
        edges=[
            GraphEdge(
                id="project:project-atlas|has_risk|risk:data-residency-issue",
                source_node_id=project.id,
                target_node_id=risk.id,
                relation_type="has_risk",
                raw_relation="Project Atlas has Data residency issue.",
                confidence=0.9,
                source_refs=[_source_ref("Project Atlas has Data residency issue.")],
            )
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
