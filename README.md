# NextSearch

NextSearch is an early-stage Python project for building a
knowledge-grounded search assistant over private, user-provided documents.

The long-term goal is not a classic chunk-only RAG system. The intended system
uses documents as the source of truth, while metadata, Markdown structure,
summaries, embeddings, and knowledge graph facts act as navigation aids. An
agent should search those aids, inspect the strongest source material, and only
then answer with provenance.

## Current Status

Implemented today:

- Provider-registry LLM service with Azure OpenAI v1 support.
- TOML and `.env` based LLM configuration.
- Text extraction from PDFs with selectable text.
- LLM-assisted PDF-to-Markdown extraction with required page anchors.
- Markdown batch stitching and artifact writing.
- Markdown section splitting with page range tracking.
- Knowledge graph extraction from Markdown sections.
- Graph node and edge models with source references.
- LLM-adjudicated graph node deduplication with deterministic and
  embedding-assisted candidate generation.
- Incremental PDF ingestion into a corpus-level graph, including document
  replacement and unchanged-document skipping by content hash.
- JSON artifacts for Markdown extraction, graph extraction, relation type
  proposals, and graph merge decisions.
- JSON-backed query-time graph retrieval and source evidence lookup.
- Provider-neutral LangGraph query agent with a controlled graph-search loop.
- Unit tests covering LLM config/service/provider behavior, PDF ingestion,
  Markdown extraction, section splitting, graph extraction, graph dedupe, and
  graph merge behavior, retrieval, and query-agent orchestration.

Not implemented yet:

- A user-facing app, API server, or production CLI.
- Vector index or database-backed retrieval store.
- Citation rendering in a user-facing interface.
- OCR for scanned or image-only PDFs.

## Requirements

- Python 3.13 or newer.
- `uv` for dependency management.
- Azure OpenAI credentials for LLM-backed extraction.

Install dependencies:

```bash
uv sync
```

Configure Azure OpenAI credentials:

```bash
cp .env.example .env
```

Then set:

```text
AZURE_OPENAI_BASE_URL=https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/
AZURE_OPENAI_API_KEY=...
```

Model routing lives in `config/llm.toml`. The configured task names currently
include:

- `markdown_extraction`
- `graph_extraction`
- `summarization`
- `query_planning`
- `graph_search_decision`
- `answer_generation`

Providers are configured under `[llm.providers.*]`, with `default_provider`
selecting the provider used by the current task routing.

## Usage

There is no supported command-line entry point yet. Use the package functions
directly from Python.

### Extract Markdown From A PDF

```python
from pathlib import Path

from nextsearch.ingestion import extract_pdf_to_markdown
from nextsearch.llm import LLMService

llm = LLMService.from_config_file()

document = extract_pdf_to_markdown(
    Path("documents/source.pdf"),
    llm,
    output_dir=Path("artifacts/source"),
)

print(document.markdown)
```

When `output_dir` is provided, the pipeline writes:

```text
artifacts/source/
  document.md
  manifest.json
  batches/
    batch-0001.output.md
```

PDF limitations:

- PDFs must contain extractable text.
- Empty pages and scanned/image-only pages are rejected.

### Split Markdown Into Sections

```python
from nextsearch.ingestion.markdown import split_markdown_into_sections

sections = split_markdown_into_sections(document)

for section in sections:
    print(section.id, section.heading, section.page_start, section.page_end)
```

Sections preserve heading context and page ranges where page anchors are
available.

### Extract A Knowledge Graph From Markdown

```python
from nextsearch.ingestion.graph import extract_knowledge_graph_from_markdown

graph = extract_knowledge_graph_from_markdown(
    document,
    llm,
    document_id="source",
    content_hash="replace-with-real-content-hash",
)

print(graph.model_dump(mode="json"))
```

The graph contains:

- Nodes for extracted entities and concepts.
- Edges for typed relationships.
- Source references with document ID, section ID, heading, page range, and quote.

### Extract A Knowledge Graph From A PDF

```python
from pathlib import Path

from nextsearch.ingestion import extract_pdf_to_knowledge_graph
from nextsearch.llm import LLMService

llm = LLMService.from_config_file()

graph = extract_pdf_to_knowledge_graph(
    Path("documents/source.pdf"),
    llm,
    document_id="source",
    output_dir=Path("artifacts/source"),
)
```

This writes `graph.json` and `relation_type_proposals.json` alongside the
Markdown artifacts when `output_dir` is provided. The graph `content_hash` is
computed from the PDF bytes. Relation type proposals are kept separate from the
canonical graph so suggested labels can be reviewed before promotion.

### Dedupe Graph Nodes

```python
from nextsearch.ingestion.graph.dedupe import dedupe_knowledge_graph

result = dedupe_knowledge_graph(graph, llm)

deduped_graph = result.graph
node_replacements = result.node_id_replacements
```

Graph dedupe generates candidate node pairs using deterministic name and
neighbor heuristics plus embedding similarity, then asks the configured LLM to
adjudicate whether each candidate refers to the same real-world entity.
Accepted merges rewrite node IDs, combine source references, and drop self-loop
edges created by the merge.

### Ingest A PDF Into A Corpus Graph

```python
from pathlib import Path

from nextsearch.ingestion import ingest_pdf_to_corpus_graph
from nextsearch.llm import LLMService

llm = LLMService.from_config_file()

corpus_graph = ingest_pdf_to_corpus_graph(
    pdf_path=Path("documents/source.pdf"),
    document_id="source",
    llm=llm,
    corpus_graph=None,
    output_dir=Path("artifacts"),
)
```

Corpus ingestion writes per-document Markdown, graph, and relation proposal
artifacts under `artifacts/documents/`, then writes the merged corpus graph and
corpus-level relation proposal artifact under `artifacts/corpus/`. When graph
dedupe makes decisions, it also writes `graph_merge_decisions.json` under
`artifacts/corpus/`. When a matching document hash is already present, the
existing corpus graph is returned unchanged.

### Query A Corpus Graph

```python
from pathlib import Path

from nextsearch.agent import build_query_agent
from nextsearch.llm import LLMService
from nextsearch.retrieval import JsonGraphStore, MarkdownArtifactSourceStore

llm = LLMService.from_config_file()
artifacts_root = Path("artifacts")

agent = build_query_agent(
    llm=llm,
    graph_store=JsonGraphStore.from_artifact_dir(artifacts_root / "corpus"),
    source_store=MarkdownArtifactSourceStore(artifacts_root),
)

result = agent.invoke({"query": "What risks are related to Project Atlas?"})

print(result["answer"])
print(result["citations"])
```

## Project Model

The source document store should remain the source of truth.

Indexes, summaries, embeddings, metadata, and graph facts are retrieval aids.
They help the agent decide what to inspect, but they should not replace source
verification.

Target query flow:

```text
User query
  -> understand intent and key entities/concepts
  -> search metadata, summaries, embeddings, and graph relationships
  -> rank candidate documents, sections, and evidence spans
  -> inspect selected source material
  -> answer with citations
```

The implemented ingestion flow is:

```text
PDF
  -> text extraction
  -> Markdown with page anchors
  -> addressable sections
  -> graph extraction with source references
  -> relation type proposal artifact
  -> corpus merge and incremental graph dedupe
```

The implemented query-agent flow is:

```text
User query
  -> LLM query planning
  -> JSON graph search
  -> source evidence lookup from Markdown artifacts
  -> LLM decision to answer or search again
  -> answer with citations
```

## Knowledge Graph Role

The knowledge graph should guide search, not replace document reading.

Graph nodes can represent entities and concepts such as people, organizations,
locations, projects, products, systems, documents, requirements, risks, events,
metrics, and dates.

Graph edges can represent relationships such as `depends_on`, `owned_by`,
`mentions`, `causes`, `contradicts`, `supports`, `impacts`,
`has_requirement`, or `has_risk`.

Every node and edge should carry source references so graph-derived claims can
be traced back to document evidence.

Example:

```text
Project Atlas
  -> depends_on -> Vendor A
  -> has_risk -> Data residency issue
  -> mentions -> Doc 12, Doc 19
```

When a user asks about `Project Atlas`, the graph can surface related entities
and documents. The query agent then inspects source evidence before answering.

## Evidence And Provenance

Current source references attach graph facts to Markdown sections and page
ranges. The long-term direction is to support smaller evidence spans where
needed.

Possible hierarchy:

```text
Document
  -> sections
  -> pages
  -> paragraphs
  -> evidence spans
```

Example target reference:

```text
Document: contract_2026.pdf
Section: 4.2 Data Residency
Paragraph: 3
Text span: character 18320-18790

Extracted edge:
Vendor A -> located_in -> Singapore

Evidence:
contract_2026.pdf, section 4.2, paragraph 3
```

## Tests

Run targeted tests with `unittest`:

```bash
uv run python -m unittest tests.test_ingestion_pdf
uv run python -m unittest tests.test_ingestion_markdown
uv run python -m unittest tests.test_ingestion_sections
uv run python -m unittest tests.test_ingestion_graph
uv run python -m unittest tests.test_llm_config
uv run python -m unittest tests.test_llm_service
uv run python -m unittest tests.test_azure_openai_v1
uv run python -m unittest tests.test_graph_dedupe
uv run python -m unittest tests.test_graph_merge
uv run python -m unittest tests.test_retrieval
uv run python -m unittest tests.test_agent
```

Use the full suite command when you want broader coverage:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

## Roadmap

Near-term:

- Add durable storage for documents, Markdown, sections, graph data, and
  extraction metadata.
- Add embeddings for documents, sections, or metadata.

Next:

- Rank candidate sources before source inspection.
- Add retrieval over metadata, summaries, embeddings, and sections.
- Improve ranking and citation rendering for grounded answers.

Later:

- Add permission-aware retrieval.
- Add re-indexing and graph refresh flows.
- Add evaluation cases for source accuracy and citation quality.
- Add observability for search steps, extraction cost, and source usage.
