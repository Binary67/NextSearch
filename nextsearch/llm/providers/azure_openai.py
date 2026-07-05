from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from nextsearch.llm.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMProviderError,
    LLMResponse,
    LLMStructuredOutputError,
    ResponseModelT,
    StructuredGenerationRequest,
    TextGenerationRequest,
)


class AzureOpenAIProvider:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        client_factory: Callable[..., Any] = OpenAI,
    ) -> None:
        self.name = name
        self._client = client_factory(api_key=api_key, base_url=base_url)

    def generate_text(self, request: TextGenerationRequest) -> LLMResponse:
        kwargs = self._text_request_kwargs(request)
        try:
            response = self._client.responses.create(**kwargs)
        except Exception as exc:
            raise LLMProviderError(
                f"Azure OpenAI text generation failed for role {request.role!r}"
            ) from exc

        text = getattr(response, "output_text", None)
        if not isinstance(text, str):
            raise LLMProviderError("Azure OpenAI response did not include output_text")

        return LLMResponse(
            text=text,
            provider=self.name,
            model=request.model,
            usage=_usage_dict(response),
        )

    def generate_json(
        self, request: StructuredGenerationRequest[ResponseModelT]
    ) -> ResponseModelT:
        kwargs = self._text_request_kwargs(request)
        kwargs["text_format"] = request.response_model

        try:
            response = self._client.responses.parse(**kwargs)
        except Exception as exc:
            raise LLMStructuredOutputError(
                f"Azure OpenAI structured generation failed for role {request.role!r}"
            ) from exc

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise LLMStructuredOutputError(
                "Azure OpenAI response did not include parsed structured output"
            )

        if isinstance(parsed, request.response_model):
            return parsed

        try:
            return request.response_model.model_validate(parsed)
        except ValidationError as exc:
            raise LLMStructuredOutputError(
                "Azure OpenAI returned invalid structured output"
            ) from exc

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            response = self._client.embeddings.create(
                model=request.model,
                input=list(request.texts),
            )
        except Exception as exc:
            raise LLMProviderError(
                f"Azure OpenAI embedding failed for role {request.role!r}"
            ) from exc

        embeddings = [list(item.embedding) for item in response.data]
        return EmbeddingResponse(
            embeddings=embeddings,
            provider=self.name,
            model=request.model,
            usage=_usage_dict(response),
        )

    def _text_request_kwargs(
        self, request: TextGenerationRequest | StructuredGenerationRequest[Any]
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": [message.to_api_input() for message in request.messages],
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        return kwargs


def _usage_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "to_dict"):
        return usage.to_dict()
    return None
