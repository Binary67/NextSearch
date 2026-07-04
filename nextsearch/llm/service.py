from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from nextsearch.llm.config import LLMConfig, ProviderConfig, load_llm_config, require_env
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
        _, provider_config, provider = self._provider_for_role(role)
        request = TextGenerationRequest(
            role=role,
            model=provider_config.text_model,
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
        _, provider_config, provider = self._provider_for_role(role)
        request = StructuredGenerationRequest(
            role=role,
            model=provider_config.text_model,
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
        _, provider_config, provider = self._provider_for_role(role)
        request = EmbeddingRequest(
            role=role,
            model=provider_config.embedding_model,
            texts=texts,
        )
        return provider.embed(request)

    def _provider_for_role(
        self, role: str
    ) -> tuple[str, ProviderConfig, LLMProvider]:
        provider_name = self._config.provider_name_for_role(role)
        provider_config = self._config.providers[provider_name]

        provider = self._providers.get(provider_name)
        if provider is None:
            provider = self._build_provider(provider_name, provider_config)
            self._providers[provider_name] = provider

        return provider_name, provider_config, provider

    def _build_provider(
        self, provider_name: str, provider_config: ProviderConfig
    ) -> LLMProvider:
        if provider_config.provider == "azure_openai_v1":
            return AzureOpenAIV1Provider(
                name=provider_name,
                base_url=require_env(provider_config.base_url_env),
                api_key=require_env(provider_config.api_key_env),
            )

        raise LLMConfigError(f"Unsupported LLM provider {provider_config.provider!r}")
