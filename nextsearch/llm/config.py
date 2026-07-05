from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal, Self

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nextsearch.llm.types import LLMConfigError

ModelTier = Literal["fast", "flagship"]


class AzureOpenAIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["azure_openai"]
    base_url_env: str
    api_key_env: str
    embedding_model: str


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fast: str
    flagship: str


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_provider: str
    providers: dict[str, AzureOpenAIConfig]
    models: ModelConfig
    tasks: dict[str, ModelTier] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_default_provider(self) -> Self:
        if self.default_provider not in self.providers:
            raise ValueError(
                f"Default LLM provider {self.default_provider!r} is not configured"
            )
        return self

    def provider_config(
        self,
        provider_name: str | None = None,
    ) -> AzureOpenAIConfig:
        name = provider_name or self.default_provider
        try:
            return self.providers[name]
        except KeyError as exc:
            raise LLMConfigError(f"LLM provider {name!r} is not configured") from exc

    def text_model_for_task(self, task: str) -> str:
        try:
            tier = self.tasks[task]
        except KeyError as exc:
            raise LLMConfigError(f"LLM task {task!r} is not configured") from exc
        return getattr(self.models, tier)


def load_llm_config(
    config_path: str | Path = "config/llm.toml",
    *,
    env_path: str | Path | None = ".env",
) -> LLMConfig:
    if env_path is not None:
        load_dotenv(dotenv_path=env_path, override=True)

    path = Path(config_path)
    try:
        raw_config = tomllib.loads(path.read_text())
    except FileNotFoundError as exc:
        raise LLMConfigError(f"LLM config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise LLMConfigError(f"Invalid LLM TOML config: {exc}") from exc

    try:
        llm_section = raw_config["llm"]
    except KeyError as exc:
        raise LLMConfigError("LLM config is missing [llm] section") from exc

    try:
        return LLMConfig.model_validate(llm_section)
    except ValidationError as exc:
        raise LLMConfigError(f"Invalid LLM config: {exc}") from exc


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise LLMConfigError(f"Required environment variable {name!r} is not set")
    return value
