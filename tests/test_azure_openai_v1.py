import unittest
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from nextsearch.llm.providers.azure_openai_v1 import AzureOpenAIV1Provider
from nextsearch.llm.types import (
    EmbeddingRequest,
    LLMMessage,
    StructuredGenerationRequest,
    TextGenerationRequest,
)


class ParsedAnswer(BaseModel):
    value: str


class FakeResponses:
    def __init__(self) -> None:
        self.create_kwargs: dict[str, Any] | None = None
        self.parse_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.create_kwargs = kwargs
        return SimpleNamespace(output_text="generated", usage={"output_tokens": 3})

    def parse(self, **kwargs: Any) -> Any:
        self.parse_kwargs = kwargs
        return SimpleNamespace(output_parsed=ParsedAnswer(value="parsed"))


class FakeEmbeddings:
    def __init__(self) -> None:
        self.create_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.create_kwargs = kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])],
            usage={"prompt_tokens": 2},
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()
        self.embeddings = FakeEmbeddings()


class AzureOpenAIV1ProviderTests(unittest.TestCase):
    def test_client_factory_receives_base_url_and_api_key(self) -> None:
        calls: list[dict[str, str]] = []

        def factory(**kwargs: str) -> FakeClient:
            calls.append(kwargs)
            return FakeClient()

        AzureOpenAIV1Provider(
            name="azure",
            base_url="https://example.openai.azure.com/openai/v1/",
            api_key="secret",
            client_factory=factory,
        )

        self.assertEqual(
            calls,
            [
                {
                    "api_key": "secret",
                    "base_url": "https://example.openai.azure.com/openai/v1/",
                }
            ],
        )

    def test_generate_text_uses_responses_create_with_text_model(self) -> None:
        client = FakeClient()
        provider = AzureOpenAIV1Provider(
            name="azure",
            base_url="https://example.openai.azure.com/openai/v1/",
            api_key="secret",
            client_factory=lambda **_: client,
        )

        response = provider.generate_text(
            TextGenerationRequest(
                role="summarization",
                model="nextsearch-chat",
                messages=[LLMMessage(role="user", content="Hello")],
            )
        )

        self.assertEqual(response.text, "generated")
        self.assertEqual(client.responses.create_kwargs["model"], "nextsearch-chat")
        self.assertEqual(
            client.responses.create_kwargs["input"],
            [{"role": "user", "content": "Hello"}],
        )

    def test_generate_json_uses_responses_parse_with_pydantic_model(self) -> None:
        client = FakeClient()
        provider = AzureOpenAIV1Provider(
            name="azure",
            base_url="https://example.openai.azure.com/openai/v1/",
            api_key="secret",
            client_factory=lambda **_: client,
        )

        result = provider.generate_json(
            StructuredGenerationRequest(
                role="graph_extraction",
                model="nextsearch-chat",
                messages=[LLMMessage(role="user", content="Extract")],
                response_model=ParsedAnswer,
            )
        )

        self.assertEqual(result, ParsedAnswer(value="parsed"))
        self.assertEqual(client.responses.parse_kwargs["model"], "nextsearch-chat")
        self.assertIs(client.responses.parse_kwargs["text_format"], ParsedAnswer)

    def test_embed_uses_embeddings_create_with_embedding_model(self) -> None:
        client = FakeClient()
        provider = AzureOpenAIV1Provider(
            name="azure",
            base_url="https://example.openai.azure.com/openai/v1/",
            api_key="secret",
            client_factory=lambda **_: client,
        )

        response = provider.embed(
            EmbeddingRequest(
                role="document_embedding",
                model="nextsearch-embed",
                texts=["Document text"],
            )
        )

        self.assertEqual(response.embeddings, [[0.1, 0.2]])
        self.assertEqual(client.embeddings.create_kwargs["model"], "nextsearch-embed")
        self.assertEqual(client.embeddings.create_kwargs["input"], ["Document text"])


if __name__ == "__main__":
    unittest.main()
