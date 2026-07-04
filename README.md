# NextSearch

NextSearch is an early-stage project for exploring a search system that combines
agentic document search with an optional knowledge graph over user-provided
documents and information.

The goal is not to build a classic chunk-only RAG system. The intended direction
is a system where an agent can search metadata, summaries, source references,
and graph relationships, then selectively inspect the original source documents
or relevant sections before answering.

## Project Objective

Build a knowledge-grounded search assistant for private/user-provided documents.

The assistant should be able to:

- Understand a user's question.
- Find relevant documents, entities, concepts, and relationships.
- Use metadata and semantic search to identify likely sources.
- Use a knowledge graph to discover connected context when relationship-aware
  reasoning is useful.
- Read the original source document or precise source section before making a
  claim.
- Answer with clear provenance and citations back to the source material.

## Core Mental Model

The source document store is the source of truth.

Indexes, embeddings, metadata, summaries, and graph nodes are navigation aids.
They help the agent decide what to inspect, but they should not replace source
verification.

```text
User query
  -> understand intent and key entities/concepts
  -> search metadata / summaries / embeddings
  -> optionally search knowledge graph nodes and relationships
  -> rank candidate documents, sections, and evidence spans
  -> agent reads selected source material
  -> answer with citations
```

## Agentic Search

In this project, agentic search means retrieval controlled by an agent rather
than a single fixed retrieval step.

Instead of:

```text
query -> vector search over chunks -> top chunks -> answer
```

the target flow is closer to:

```text
query
  -> plan what information is needed
  -> search document metadata and summaries
  -> inspect relevant sections or full documents
  -> search again if evidence is incomplete
  -> compare sources
  -> answer with citations
```

Agentic search can still use embeddings, chunking, summaries, and indexes, but
the agent decides which tools to use and when enough evidence has been gathered.

## Knowledge Graph Role

The knowledge graph should guide search, not replace document reading.

The graph can contain:

- Nodes for entities, concepts, documents, projects, people, organizations,
  policies, risks, requirements, and events.
- Edges for relationships such as `depends_on`, `owned_by`, `mentioned_in`,
  `causes`, `contradicts`, `supports`, `impacts`, or `related_to`.
- Provenance links from every node or edge back to source evidence.

Example:

```text
Project Atlas
  -> depends_on -> Vendor A
  -> has_risk -> Data residency issue
  -> mentioned_in -> Doc 12, Doc 19
```

When a user asks about `Project Atlas`, the graph can surface related entities
and documents. The agent should then inspect the strongest source evidence
before answering.

The graph should not cause the system to blindly read every connected document.
Candidate sources should be ranked by relevance, relationship type, confidence,
recency, authority, permissions, and source quality.

## Evidence Spans

The system will likely need smaller addressable units inside documents, but not
only for classic chunk-based RAG.

Smaller units are needed so graph facts and answers can point back to precise
source evidence.

Possible hierarchy:

```text
Document
  -> sections
  -> pages
  -> paragraphs
  -> evidence spans
```

Example:

```text
Document: contract_2026.pdf
Section: 4.2 Data Residency
Paragraph: 3
Text span: character 18320-18790

Extracted edge:
Vendor A -> stores_data_in -> Singapore

Evidence:
contract_2026.pdf, section 4.2, paragraph 3
```

These spans are evidence anchors. They let the agent jump to the right part of a
document, verify extracted graph facts, and cite claims.

## Suggested Architecture

One ingestion pipeline should produce multiple derived indexes from the same
canonical document store.

```text
User documents
  -> text extraction
  -> canonical document store
  -> document metadata
  -> document and section summaries
  -> document / section / metadata embeddings
  -> evidence spans
  -> entity and relationship extraction
  -> knowledge graph
```

At query time:

```text
User query
  -> agent decides retrieval strategy
  -> metadata / vector search
  -> optional graph lookup and graph expansion
  -> candidate source ranking
  -> source reading and verification
  -> answer generation with citations
```

## Suggested Data Model

Initial conceptual tables or collections:

```text
documents
- id
- title
- file_type
- source_path
- created_at
- full_text
- metadata
- summary

document_sections
- id
- document_id
- heading
- page_start
- page_end
- text
- embedding

evidence_spans
- id
- document_id
- section_id
- page
- paragraph_index
- char_start
- char_end
- text

graph_nodes
- id
- type
- name
- description
- source_span_ids

graph_edges
- id
- source_node_id
- target_node_id
- relation_type
- confidence
- source_span_ids
```

## Design Principles

- Keep source documents as the source of truth.
- Treat metadata, summaries, embeddings, and graph facts as retrieval aids.
- Every graph node and edge should have provenance.
- Prefer selective source reading over reading every connected document.
- Start with agentic document search first.
- Add the knowledge graph when relationship-heavy questions become important.
- Keep the first implementation small and directly testable.

## To Do

### Phase 0: Product and Data Design

- Define the first target document types.
- Define the canonical document model.
- Decide what counts as a document, section, paragraph, and evidence span.
- Decide what metadata should be generated for each document.
- Decide whether citations should point to pages, paragraphs, character spans,
  or all of them.

### Phase 1: Agentic Document Search MVP

- Add document ingestion.
- Extract full text from uploaded or local documents.
- Generate document-level metadata and summaries.
- Split documents into addressable sections and evidence spans.
- Add embeddings for documents, sections, and/or metadata.
- Implement query-time retrieval over metadata and summaries.
- Let the agent inspect selected source sections or full documents.
- Return answers with citations.

### Phase 2: Knowledge Graph Prototype

- Extract entities from documents and evidence spans.
- Extract relationships between entities.
- Store graph nodes and edges with confidence scores.
- Attach source evidence spans to every node and edge.
- Implement graph lookup from a user query.
- Implement graph expansion from matched nodes to related nodes and documents.
- Rank graph-connected source documents before agent inspection.

### Phase 3: Hybrid Retrieval

- Combine vector search, metadata search, and graph lookup.
- Let the agent choose the retrieval strategy based on question type.
- Support relationship-aware questions across multiple documents.
- Support global corpus questions such as themes, risks, dependencies, or
  contradictions.
- Add evaluation cases for source accuracy and citation quality.

### Phase 4: Hardening

- Add permission-aware retrieval.
- Track extraction confidence and stale metadata.
- Add re-indexing and graph refresh flows.
- Add tests for ingestion, retrieval, graph provenance, and citation grounding.
- Add observability for agent search steps and source usage.

