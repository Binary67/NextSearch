from __future__ import annotations

from nextsearch.llm.config import AzureOpenAIConfig, require_env
from nextsearch.llm.providers.azure_openai import AzureOpenAIProvider
from nextsearch.llm.types import LLMProvider


def build_llm_provider(
    *,
    name: str,
    provider_config: AzureOpenAIConfig,
) -> LLMProvider:
    return AzureOpenAIProvider(
        name=name,
        base_url=require_env(provider_config.base_url_env),
        api_key=require_env(provider_config.api_key_env),
    )
