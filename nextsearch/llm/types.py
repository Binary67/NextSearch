from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, Sequence, TypeVar

from pydantic import BaseModel


MessageRole = Literal["system", "user", "assistant", "developer"]
ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM setup and provider failures."""


class LLMConfigError(LLMError):
    """Raised when LLM configuration or credentials are invalid."""


class LLMProviderError(LLMError):
    """Raised when an LLM provider call fails."""


class LLMStructuredOutputError(LLMProviderError):
    """Raised when structured LLM output cannot be validated."""


@dataclass(frozen=True)
class LLMMessage:
    role: MessageRole
    content: str

    def to_api_input(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class TextGenerationRequest:
    role: str
    model: str
    messages: Sequence[LLMMessage]
    temperature: float | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class StructuredGenerationRequest(Generic[ResponseModelT]):
    role: str
    model: str
    messages: Sequence[LLMMessage]
    response_model: type[ResponseModelT]
    temperature: float | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class EmbeddingRequest:
    role: str
    model: str
    texts: Sequence[str]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class EmbeddingResponse:
    embeddings: list[list[float]]
    provider: str
    model: str
    usage: dict[str, Any] | None = None


class LLMProvider(Protocol):
    name: str

    def generate_text(self, request: TextGenerationRequest) -> LLMResponse:
        ...

    def generate_json(
        self, request: StructuredGenerationRequest[ResponseModelT]
    ) -> ResponseModelT | dict[str, Any]:
        ...

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        ...
