from __future__ import annotations

from nextsearch.ingestion.graph.models import ENTITY_TYPES, RELATION_TYPES
from nextsearch.ingestion.models import DocumentSection
from nextsearch.llm.types import LLMMessage


SYSTEM_PROMPT = """You extract a small knowledge graph from document sections.
Extract only explicit facts supported by the section text.
Do not create a relationship for every sentence.
Skip trivial, repeated, vague, or unsupported relationships.
Every node and edge must include source evidence with a direct quote.
Use only the allowed entity types and relation types."""


def build_graph_extraction_messages(section: DocumentSection) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(role="user", content=_build_user_prompt(section)),
    ]


def _build_user_prompt(section: DocumentSection) -> str:
    entity_types = ", ".join(ENTITY_TYPES)
    relation_types = ", ".join(RELATION_TYPES)
    heading_path = " > ".join(section.heading_path)
    return (
        "Extract meaningful graph nodes and relationships from this document section.\n"
        f"Allowed entity types: {entity_types}\n"
        f"Allowed relation types: {relation_types}\n\n"
        "For every source_ref, use this exact metadata and add the supporting quote:\n"
        f"section_id: {section.id}\n"
        f"heading: {section.heading}\n"
        f"page_start: {section.page_start}\n"
        f"page_end: {section.page_end}\n\n"
        "Use raw_relation for the wording in the source text, while relation_type must "
        "come from the allowed list.\n\n"
        f"Heading path: {heading_path}\n\n"
        "Section text:\n"
        f"{section.text}"
    )
