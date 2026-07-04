from __future__ import annotations

from nextsearch.llm.config import AzureOpenAIV1Config, require_env
from nextsearch.llm.providers.azure_openai_v1 import AzureOpenAIV1Provider
from nextsearch.llm.types import LLMProvider


def build_llm_provider(
    *,
    name: str,
    provider_config: AzureOpenAIV1Config,
) -> LLMProvider:
    return AzureOpenAIV1Provider(
        name=name,
        base_url=require_env(provider_config.base_url_env),
        api_key=require_env(provider_config.api_key_env),
    )
