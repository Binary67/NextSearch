from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from nextsearch.llm.config import LLMConfig, load_llm_config
from nextsearch.llm.factory import build_llm_provider
from nextsearch.llm.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMStructuredOutputError,
    ResponseModelT,
    StructuredGenerationRequest,
    TextGenerationRequest,
)


class LLMService:
    def __init__(
        self,
        config: LLMConfig,
        *,
        providers: Mapping[str, LLMProvider] | None = None,
    ) -> None:
        self._config = config
        self._providers = dict(providers or {})

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path = "config/llm.toml",
        *,
        env_path: str | Path | None = ".env",
    ) -> LLMService:
        return cls(load_llm_config(config_path, env_path=env_path))

    def generate_text(
        self,
        *,
        role: str,
        messages: Sequence[LLMMessage],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        model = self._config.text_model_for_task(role)
        provider = self._provider()
        request = TextGenerationRequest(
            role=role,
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return provider.generate_text(request)

    def generate_json(
        self,
        *,
        role: str,
        messages: Sequence[LLMMessage],
        response_model: type[ResponseModelT],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> ResponseModelT:
        model = self._config.text_model_for_task(role)
        provider = self._provider()
        request = StructuredGenerationRequest(
            role=role,
            model=model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        result = provider.generate_json(request)
        if isinstance(result, response_model):
            return result

        try:
            return response_model.model_validate(result)
        except ValidationError as exc:
            raise LLMStructuredOutputError(
                f"Provider returned invalid structured output for {response_model.__name__}"
            ) from exc

    def embed(self, *, role: str, texts: Sequence[str]) -> EmbeddingResponse:
        provider = self._provider()
        provider_config = self._config.provider_config()
        request = EmbeddingRequest(
            role=role,
            model=provider_config.embedding_model,
            texts=texts,
        )
        return provider.embed(request)

    def embedding_provider_name(self) -> str:
        return self._config.default_provider

    def embedding_model(self) -> str:
        return self._config.provider_config().embedding_model

    def _provider(self, provider_name: str | None = None) -> LLMProvider:
        name = provider_name or self._config.default_provider
        provider = self._providers.get(name)
        if provider is None:
            provider = build_llm_provider(
                name=name,
                provider_config=self._config.provider_config(name),
            )
            self._providers[name] = provider

        return provider
