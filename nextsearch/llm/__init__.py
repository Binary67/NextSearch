"""Provider-neutral LLM access for NextSearch."""

from nextsearch.llm.config import LLMConfig, load_llm_config
from nextsearch.llm.service import LLMService
from nextsearch.llm.types import (
    EmbeddingResponse,
    LLMConfigError,
    LLMError,
    LLMMessage,
    LLMProviderError,
    LLMResponse,
    LLMStructuredOutputError,
)

__all__ = [
    "EmbeddingResponse",
    "LLMConfig",
    "LLMConfigError",
    "LLMError",
    "LLMMessage",
    "LLMProviderError",
    "LLMResponse",
    "LLMService",
    "LLMStructuredOutputError",
    "load_llm_config",
]
