import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from nextsearch.ingestion.artifacts import (
    write_graph_artifact,
    write_graph_merge_decisions_artifact,
    write_relation_type_proposals_artifact,
)
from nextsearch.ingestion.graph.dedupe import (
    dedupe_knowledge_graph,
    dedupe_knowledge_graph_incremental,
)
from nextsearch.ingestion.graph.llm_extractor import normalize_edge_id, normalize_node_id
from nextsearch.ingestion.graph.models import (
    GraphEdge,
    GraphDedupeResult,
    GraphNode,
    GraphNodeMergeDecision,
    KnowledgeGraph,
    SourceRef,
)
from nextsearch.ingestion.graph.relation_proposals import build_relation_type_proposals
from nextsearch.llm.types import EmbeddingResponse, LLMMessage


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


class FakeRelationProposalLLM:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
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
        return response_model.model_validate(self.output)


class KnowledgeGraphDedupeTests(unittest.TestCase):
    def test_dedupe_merges_vendor_alias_and_rewrites_edges(self) -> None:
        project = _node("project", "Project Atlas", "Project Atlas uses Vendor A.")
        vendor = _node("organization", "Vendor A", "Vendor A provides hosting.")
        vendor_ltd = _node(
            "organization",
            "Vendor A Ltd.",
            "Vendor A Ltd. signed the hosting agreement.",
        )
        singapore = _node("location", "Singapore", "Singapore is the hosting region.")
        graph = _graph(
            nodes=[project, vendor, vendor_ltd, singapore],
            edges=[
                _edge(project.id, "depends_on", vendor.id, "Project Atlas depends on Vendor A.", 0.7),
                _edge(
                    project.id,
                    "depends_on",
                    vendor_ltd.id,
                    "Project Atlas depends on Vendor A Ltd. for hosting.",
                    0.9,
                ),
                _edge(vendor.id, "located_in", singapore.id, "Vendor A is located in Singapore.", 0.8),
            ],
        )
        llm = FakeDedupeLLM(
            [{"decision": "same", "confidence": 0.95, "reason": "Legal name variant."}]
        )

        result = dedupe_knowledge_graph(graph, llm)  # type: ignore[arg-type]

        self.assertEqual(
            result.node_id_replacements,
            {"organization:vendor-a": "organization:vendor-a-ltd"},
        )
        self.assertEqual(len(result.graph.nodes), 3)
        canonical = _node_by_id(result.graph, "organization:vendor-a-ltd")
        self.assertEqual(canonical.name, "Vendor A Ltd.")
        self.assertEqual(canonical.aliases, ["Vendor A"])
        self.assertEqual(len(canonical.source_refs), 2)
        self.assertEqual(len(result.graph.edges), 2)

        depends_on = _edge_by_id(
            result.graph,
            "project:project-atlas|depends_on|organization:vendor-a-ltd",
        )
        self.assertEqual(depends_on.confidence, 0.9)
        self.assertEqual(len(depends_on.source_refs), 2)

        located_in = _edge_by_id(
            result.graph,
            "organization:vendor-a-ltd|located_in|location:singapore",
        )
        self.assertEqual(located_in.source_node_id, "organization:vendor-a-ltd")
        self.assertEqual(llm.calls[0]["role"], "graph_extraction")
        self.assertEqual(llm.calls[0]["temperature"], 0)
        self.assertEqual(result.merge_decisions[0].canonical_node_id, "organization:vendor-a-ltd")

    def test_dedupe_does_not_merge_uncertain_candidate(self) -> None:
        vendor = _node("organization", "Vendor A", "Vendor A is one vendor.")
        vendor_ltd = _node(
            "organization",
            "Vendor A Ltd.",
            "Vendor A Ltd. is mentioned separately.",
        )
        graph = _graph(nodes=[vendor, vendor_ltd], edges=[])
        llm = FakeDedupeLLM(
            [{"decision": "uncertain", "confidence": 0.95, "reason": "Evidence is ambiguous."}]
        )

        result = dedupe_knowledge_graph(graph, llm)  # type: ignore[arg-type]

        self.assertEqual(result.node_id_replacements, {})
        self.assertEqual({node.id for node in result.graph.nodes}, {vendor.id, vendor_ltd.id})
        self.assertIsNone(result.merge_decisions[0].canonical_node_id)

    def test_dedupe_does_not_merge_low_confidence_same_candidate(self) -> None:
        vendor = _node("organization", "Vendor A", "Vendor A is one vendor.")
        vendor_ltd = _node("organization", "Vendor A Ltd.", "Vendor A Ltd. is related.")
        graph = _graph(nodes=[vendor, vendor_ltd], edges=[])
        llm = FakeDedupeLLM(
            [{"decision": "same", "confidence": 0.89, "reason": "Likely but not certain."}]
        )

        result = dedupe_knowledge_graph(graph, llm)  # type: ignore[arg-type]

        self.assertEqual(result.node_id_replacements, {})
        self.assertEqual(len(result.graph.nodes), 2)

    def test_dedupe_never_compares_different_node_types(self) -> None:
        project = _node("project", "Atlas", "Atlas is a project.")
        concept = _node("concept", "Atlas", "Atlas is a concept.")
        graph = _graph(nodes=[project, concept], edges=[])
        llm = FakeDedupeLLM([])

        result = dedupe_knowledge_graph(graph, llm)  # type: ignore[arg-type]

        self.assertEqual(llm.calls, [])
        self.assertEqual({node.id for node in result.graph.nodes}, {project.id, concept.id})

    def test_dedupe_removes_self_loop_edges_created_by_merge(self) -> None:
        vendor = _node("organization", "Vendor A", "Vendor A is named.")
        vendor_ltd = _node("organization", "Vendor A Ltd.", "Vendor A Ltd. is named.")
        graph = _graph(
            nodes=[vendor, vendor_ltd],
            edges=[
                _edge(vendor.id, "related_to", vendor_ltd.id, "Vendor A relates to Vendor A Ltd.", 0.8),
            ],
        )
        llm = FakeDedupeLLM(
            [{"decision": "same", "confidence": 0.97, "reason": "Same organization."}]
        )

        result = dedupe_knowledge_graph(graph, llm)  # type: ignore[arg-type]

        self.assertEqual(len(result.graph.nodes), 1)
        self.assertEqual(result.graph.edges, [])

    def test_semantic_candidate_sends_similar_nodes_to_llm(self) -> None:
        platform = _node("system", "Atlas storage platform", "Atlas stores regulated files.")
        vault = _node("system", "Data vault service", "The data vault stores regulated files.")
        graph = _graph(nodes=[platform, vault], edges=[])
        llm = FakeDedupeLLM(
            [{"decision": "same", "confidence": 0.95, "reason": "Same system."}],
            embeddings=[[[1.0, 0.0], [0.96, 0.1]]],
        )

        result = dedupe_knowledge_graph(graph, llm)  # type: ignore[arg-type]

        self.assertEqual(len(llm.calls), 1)
        self.assertIn("semantic_similarity", llm.calls[0]["messages"][1].content)
        self.assertEqual(len(result.graph.nodes), 1)

    def test_semantic_dedupe_never_compares_different_node_types(self) -> None:
        system = _node("system", "Atlas", "Atlas is the system.")
        concept = _node("concept", "Data vault", "Data vault is the concept.")
        graph = _graph(nodes=[system, concept], edges=[])
        llm = FakeDedupeLLM([])

        dedupe_knowledge_graph(graph, llm)  # type: ignore[arg-type]

        self.assertEqual(llm.embed_calls, [])
        self.assertEqual(llm.calls, [])

    def test_incremental_semantic_dedupe_skips_old_to_old_candidates(self) -> None:
        old_system = _node("system", "Atlas storage platform", "Atlas stores files.")
        old_vault = _node("system", "Data vault service", "The data vault stores files.")
        incoming = _node("system", "Billing engine", "Billing engine calculates invoices.")
        graph = _graph(nodes=[old_system, old_vault, incoming], edges=[])
        llm = FakeDedupeLLM(
            [],
            embeddings=[[[1.0, 0.0], [0.96, 0.1], [0.0, 1.0]]],
        )

        dedupe_knowledge_graph_incremental(
            graph,
            llm,  # type: ignore[arg-type]
            incoming_node_ids={incoming.id},
        )

        self.assertEqual(llm.calls, [])

    def test_write_graph_merge_decisions_artifact(self) -> None:
        graph = _graph(nodes=[], edges=[])
        result = GraphDedupeResult(
            graph=graph,
            merge_decisions=[
                GraphNodeMergeDecision(
                    node_ids=["organization:vendor-a", "organization:vendor-a-ltd"],
                    decision="same",
                    confidence=0.95,
                    reason="Legal name variant.",
                    canonical_node_id="organization:vendor-a-ltd",
                )
            ],
            node_id_replacements={"organization:vendor-a": "organization:vendor-a-ltd"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_graph_merge_decisions_artifact(result, output_dir)

            payload = json.loads(
                (output_dir / "graph_merge_decisions.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            payload["node_id_replacements"],
            {"organization:vendor-a": "organization:vendor-a-ltd"},
        )
        self.assertEqual(payload["merge_decisions"][0]["decision"], "same")

    def test_relation_type_proposal_artifact_keeps_graph_canonical(self) -> None:
        product = _node("product", "Platform A", "Platform A is certified.")
        certifier = _node("organization", "CertCo", "CertCo certifies products.")
        edge = _edge(
            product.id,
            "related_to",
            certifier.id,
            "Platform A is certified by CertCo.",
            0.9,
        ).model_copy(
            update={
                "raw_relation": "is certified by",
                "source_refs": [
                    _source_ref("Platform A is certified by CertCo."),
                    _source_ref_for_document("doc-2", "CertCo certified Platform A."),
                    _source_ref_for_document("doc-2", "The certification is active."),
                ],
            }
        )
        graph = _graph(nodes=[product, certifier], edges=[edge])
        llm = FakeRelationProposalLLM(
            {
                "proposals": [
                    {
                        "proposed_relation_type": "certified_by",
                        "closest_existing_relation_type": "related_to",
                        "raw_relations": ["hallucinated wording"],
                        "source_node_types": ["concept"],
                        "target_node_types": ["concept"],
                        "supporting_edge_ids": [edge.id],
                        "evidence_count": 1,
                        "document_count": 1,
                        "confidence": 0.91,
                        "status": "proposed",
                        "promotion_ready": False,
                        "rationale": "Recurring certification relation.",
                    }
                ]
            }
        )

        proposals = build_relation_type_proposals(graph, llm)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_graph_artifact(graph, output_dir)
            write_relation_type_proposals_artifact(proposals, output_dir)
            graph_payload = json.loads(
                (output_dir / "graph.json").read_text(encoding="utf-8")
            )
            proposal_payload = json.loads(
                (output_dir / "relation_type_proposals.json").read_text(encoding="utf-8")
            )

        self.assertEqual(graph_payload["edges"][0]["relation_type"], "related_to")
        self.assertNotIn("proposed_relation_type", graph_payload["edges"][0])
        self.assertEqual(
            proposal_payload["proposals"][0]["proposed_relation_type"],
            "certified_by",
        )
        self.assertEqual(
            proposal_payload["proposals"][0]["raw_relations"],
            ["is certified by"],
        )
        self.assertEqual(proposal_payload["proposals"][0]["evidence_count"], 3)
        self.assertEqual(proposal_payload["proposals"][0]["document_count"], 2)
        self.assertTrue(proposal_payload["proposals"][0]["promotion_ready"])


def _graph(nodes: list[GraphNode], edges: list[GraphEdge]) -> KnowledgeGraph:
    return KnowledgeGraph(
        document_id="doc-1",
        content_hash="hash-1",
        source_path="sample.pdf",
        page_count=1,
        nodes=nodes,
        edges=edges,
    )


def _node(node_type: str, name: str, quote: str) -> GraphNode:
    return GraphNode(
        id=normalize_node_id(node_type, name),
        type=node_type,  # type: ignore[arg-type]
        name=name,
        source_refs=[_source_ref(quote)],
    )


def _edge(
    source_node_id: str,
    relation_type: str,
    target_node_id: str,
    quote: str,
    confidence: float,
) -> GraphEdge:
    return GraphEdge(
        id=normalize_edge_id(source_node_id, relation_type, target_node_id),
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        relation_type=relation_type,  # type: ignore[arg-type]
        raw_relation=relation_type,
        confidence=confidence,
        source_refs=[_source_ref(quote)],
    )


def _source_ref(quote: str) -> SourceRef:
    return _source_ref_for_document("doc-1", quote)


def _source_ref_for_document(document_id: str, quote: str) -> SourceRef:
    return SourceRef(
        document_id=document_id,
        section_id="section-0001",
        heading="Overview",
        quote=quote,
    )


def _node_by_id(graph: KnowledgeGraph, node_id: str) -> GraphNode:
    return next(node for node in graph.nodes if node.id == node_id)


def _edge_by_id(graph: KnowledgeGraph, edge_id: str) -> GraphEdge:
    return next(edge for edge in graph.edges if edge.id == edge_id)


if __name__ == "__main__":
    unittest.main()
