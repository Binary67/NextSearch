class IngestionError(Exception):
    """Base exception for document ingestion failures."""


class PDFExtractionError(IngestionError):
    """Raised when PDF page text cannot be extracted."""


class MarkdownExtractionError(IngestionError):
    """Raised when LLM Markdown extraction fails."""


class GraphExtractionError(IngestionError):
    """Raised when LLM knowledge graph extraction fails."""


class GraphDedupeError(IngestionError):
    """Raised when LLM knowledge graph dedupe fails."""
