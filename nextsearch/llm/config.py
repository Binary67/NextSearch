from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nextsearch.llm.types import LLMConfigError


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["azure_openai_v1"]
    base_url_env: str
    api_key_env: str
    text_model: str
    embedding_model: str


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_provider: str
    roles: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, ProviderConfig]

    @model_validator(mode="after")
    def validate_provider_references(self) -> LLMConfig:
        if self.default_provider not in self.providers:
            raise ValueError(
                f"default_provider {self.default_provider!r} is not defined in providers"
            )

        missing = sorted(
            {
                provider_name
                for provider_name in self.roles.values()
                if provider_name not in self.providers
            }
        )
        if missing:
            raise ValueError(
                "roles reference unknown provider(s): " + ", ".join(missing)
            )

        return self

    def provider_name_for_role(self, role: str) -> str:
        try:
            return self.roles[role]
        except KeyError as exc:
            raise LLMConfigError(f"LLM role {role!r} is not configured") from exc

    def provider_for_role(self, role: str) -> ProviderConfig:
        provider_name = self.provider_name_for_role(role)
        return self.providers[provider_name]


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
