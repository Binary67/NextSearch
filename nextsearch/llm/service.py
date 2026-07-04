from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from nextsearch.llm.config import AzureConfig, LLMConfig, load_llm_config, require_env
from nextsearch.llm.providers.azure_openai_v1 import AzureOpenAIV1Provider
from nextsearch.llm.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMConfigError,
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
        provider = self._azure_provider()
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
        provider = self._azure_provider()
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
        provider = self._azure_provider()
        request = EmbeddingRequest(
            role=role,
            model=self._config.azure.embedding_model,
            texts=texts,
        )
        return provider.embed(request)

    def _azure_provider(self) -> LLMProvider:
        provider = self._providers.get("azure")
        if provider is None:
            provider = self._build_provider(self._config.azure)
            self._providers["azure"] = provider

        return provider

    def _build_provider(self, provider_config: AzureConfig) -> LLMProvider:
        if provider_config.provider == "azure_openai_v1":
            return AzureOpenAIV1Provider(
                name="azure",
                base_url=require_env(provider_config.base_url_env),
                api_key=require_env(provider_config.api_key_env),
            )

        raise LLMConfigError(f"Unsupported LLM provider {provider_config.provider!r}")
