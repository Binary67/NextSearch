class IngestionError(Exception):
    """Base exception for document ingestion failures."""


class PDFExtractionError(IngestionError):
    """Raised when PDF page text cannot be extracted."""


class MarkdownExtractionError(IngestionError):
    """Raised when LLM Markdown extraction fails."""
