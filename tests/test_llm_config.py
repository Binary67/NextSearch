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


if __name__ == "__main__":
    unittest.main()
