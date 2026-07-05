import os
import tempfile
import unittest
from pathlib import Path

from nextsearch.llm.config import LLMConfig, load_llm_config, require_env
from nextsearch.llm.types import LLMConfigError


VALID_CONFIG = """
[llm]
default_provider = "azure"

[llm.providers.azure]
provider = "azure_openai"
base_url_env = "AZURE_OPENAI_BASE_URL"
api_key_env = "AZURE_OPENAI_API_KEY"
embedding_model = "nextsearch-embed"

[llm.models]
fast = "nextsearch-chat"
flagship = "nextsearch-flagship"

[llm.tasks]
summarization = "fast"
"""

EXTRACTION_CONFIG = """
[llm]
default_provider = "azure"

[llm.providers.azure]
provider = "azure_openai"
base_url_env = "AZURE_OPENAI_BASE_URL"
api_key_env = "AZURE_OPENAI_API_KEY"
embedding_model = "nextsearch-embed"

[llm.models]
fast = "nextsearch-extraction"
flagship = "strong-agent-model"

[llm.tasks]
markdown_extraction = "fast"
answer_generation = "flagship"
"""


class LLMConfigTests(unittest.TestCase):
    def test_load_valid_toml_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "llm.toml"
            config_path.write_text(VALID_CONFIG)

            config = load_llm_config(config_path, env_path=None)

        self.assertIsInstance(config, LLMConfig)
        self.assertEqual(config.text_model_for_task("summarization"), "nextsearch-chat")
        self.assertEqual(config.provider_config().embedding_model, "nextsearch-embed")

    def test_task_referencing_unknown_model_tier_fails_validation(self) -> None:
        config = VALID_CONFIG.replace(
            'summarization = "fast"',
            'summarization = "missing_tier"',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "llm.toml"
            config_path.write_text(config)

            with self.assertRaises(LLMConfigError):
                load_llm_config(config_path, env_path=None)

    def test_missing_required_env_var_raises_config_error(self) -> None:
        name = "NEXTSEARCH_TEST_MISSING_ENV"
        old_value = os.environ.pop(name, None)
        try:
            with self.assertRaises(LLMConfigError):
                require_env(name)
        finally:
            if old_value is not None:
                os.environ[name] = old_value

    def test_missing_task_raises_config_error(self) -> None:
        config = LLMConfig.model_validate(
            {
                "default_provider": "azure",
                "providers": {
                    "azure": {
                        "provider": "azure_openai",
                        "base_url_env": "AZURE_OPENAI_BASE_URL",
                        "api_key_env": "AZURE_OPENAI_API_KEY",
                        "embedding_model": "nextsearch-embed",
                    },
                },
                "models": {
                    "fast": "nextsearch-chat",
                    "flagship": "nextsearch-flagship",
                },
                "tasks": {},
            }
        )

        with self.assertRaises(LLMConfigError):
            config.text_model_for_task("summarization")

    def test_default_provider_must_be_configured(self) -> None:
        config = VALID_CONFIG.replace(
            'default_provider = "azure"',
            'default_provider = "missing"',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "llm.toml"
            config_path.write_text(config)

            with self.assertRaises(LLMConfigError):
                load_llm_config(config_path, env_path=None)

    def test_unsupported_provider_type_fails_validation(self) -> None:
        config = VALID_CONFIG.replace(
            'provider = "azure_openai"',
            'provider = "unsupported_provider"',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "llm.toml"
            config_path.write_text(config)

            with self.assertRaises(LLMConfigError):
                load_llm_config(config_path, env_path=None)

    def test_unknown_provider_lookup_raises_config_error(self) -> None:
        config = LLMConfig.model_validate(
            {
                "default_provider": "azure",
                "providers": {
                    "azure": {
                        "provider": "azure_openai",
                        "base_url_env": "AZURE_OPENAI_BASE_URL",
                        "api_key_env": "AZURE_OPENAI_API_KEY",
                        "embedding_model": "nextsearch-embed",
                    },
                },
                "models": {
                    "fast": "nextsearch-chat",
                    "flagship": "nextsearch-flagship",
                },
                "tasks": {},
            }
        )

        with self.assertRaises(LLMConfigError):
            config.provider_config("missing")

    def test_tasks_can_use_different_model_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "llm.toml"
            config_path.write_text(EXTRACTION_CONFIG)

            config = load_llm_config(config_path, env_path=None)

        self.assertEqual(
            config.text_model_for_task("markdown_extraction"),
            "nextsearch-extraction",
        )
        self.assertEqual(
            config.text_model_for_task("answer_generation"),
            "strong-agent-model",
        )

    def test_env_file_overrides_existing_environment_value(self) -> None:
        name = "NEXTSEARCH_TEST_DOTENV_OVERRIDE"
        old_value = os.environ.get(name)
        os.environ[name] = "from-environment"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = Path(tmpdir) / "llm.toml"
                env_path = Path(tmpdir) / ".env"
                config_path.write_text(VALID_CONFIG)
                env_path.write_text(f"{name}=from-dotenv\n")

                load_llm_config(config_path, env_path=env_path)

            self.assertEqual(os.environ[name], "from-dotenv")
        finally:
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


if __name__ == "__main__":
    unittest.main()
