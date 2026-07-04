import os
import tempfile
import unittest
from pathlib import Path

from nextsearch.llm.config import LLMConfig, load_llm_config, require_env
from nextsearch.llm.types import LLMConfigError


VALID_CONFIG = """
[llm]
default_provider = "azure_primary"

[llm.roles]
summarization = "azure_primary"

[llm.providers.azure_primary]
provider = "azure_openai_v1"
base_url_env = "AZURE_OPENAI_BASE_URL"
api_key_env = "AZURE_OPENAI_API_KEY"
text_model = "nextsearch-chat"
embedding_model = "nextsearch-embed"
"""

EXTRACTION_CONFIG = """
[llm]
default_provider = "azure_primary"

[llm.roles]
markdown_extraction = "azure_extraction"
answer_generation = "azure_primary"

[llm.providers.azure_primary]
provider = "azure_openai_v1"
base_url_env = "AZURE_OPENAI_BASE_URL"
api_key_env = "AZURE_OPENAI_API_KEY"
text_model = "strong-agent-model"
embedding_model = "nextsearch-embed"

[llm.providers.azure_extraction]
provider = "azure_openai_v1"
base_url_env = "AZURE_OPENAI_BASE_URL"
api_key_env = "AZURE_OPENAI_API_KEY"
text_model = "small-extraction-model"
embedding_model = "nextsearch-embed"
"""


class LLMConfigTests(unittest.TestCase):
    def test_load_valid_toml_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "llm.toml"
            config_path.write_text(VALID_CONFIG)

            config = load_llm_config(config_path, env_path=None)

        self.assertIsInstance(config, LLMConfig)
        self.assertEqual(config.provider_name_for_role("summarization"), "azure_primary")
        self.assertEqual(
            config.provider_for_role("summarization").text_model,
            "nextsearch-chat",
        )

    def test_role_referencing_unknown_provider_fails_validation(self) -> None:
        config = VALID_CONFIG.replace(
            'summarization = "azure_primary"',
            'summarization = "missing_provider"',
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

    def test_missing_role_raises_config_error(self) -> None:
        config = LLMConfig.model_validate(
            {
                "default_provider": "azure_primary",
                "roles": {},
                "providers": {
                    "azure_primary": {
                        "provider": "azure_openai_v1",
                        "base_url_env": "AZURE_OPENAI_BASE_URL",
                        "api_key_env": "AZURE_OPENAI_API_KEY",
                        "text_model": "nextsearch-chat",
                        "embedding_model": "nextsearch-embed",
                    }
                },
            }
        )

        with self.assertRaises(LLMConfigError):
            config.provider_name_for_role("summarization")

    def test_role_can_use_different_extraction_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "llm.toml"
            config_path.write_text(EXTRACTION_CONFIG)

            config = load_llm_config(config_path, env_path=None)

        self.assertEqual(
            config.provider_for_role("markdown_extraction").text_model,
            "small-extraction-model",
        )
        self.assertEqual(
            config.provider_for_role("answer_generation").text_model,
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
