import unittest
from typing import Any

from pydantic import BaseModel

from nextsearch.llm.config import LLMConfig
from nextsearch.llm.service import LLMService
from nextsearch.llm.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMMessage,
    LLMResponse,
    LLMStructuredOutputError,
    StructuredGenerationRequest,
    TextGenerationRequest,
)


class Answer(BaseModel):
    value: str


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.json_result: Any = {"value": "ok"}

    def generate_text(self, request: TextGenerationRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text="hello", provider="azure", model=request.model)

    def generate_json(self, request: StructuredGenerationRequest[Any]) -> Any:
        self.requests.append(request)
        return self.json_result

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        return EmbeddingResponse(
            embeddings=[[0.1, 0.2]],
            provider="azure",
            model=request.model,
        )


def config() -> LLMConfig:
    return LLMConfig.model_validate(
        {
            "default_provider": "azure",
            "providers": {
                "azure": {
                    "provider": "azure_openai_v1",
                    "base_url_env": "AZURE_OPENAI_BASE_URL",
                    "api_key_env": "AZURE_OPENAI_API_KEY",
                    "embedding_model": "nextsearch-embed",
                },
            },
            "models": {
                "fast": "nextsearch-fast",
                "flagship": "nextsearch-flagship",
            },
            "tasks": {
                "summarization": "fast",
                "graph_extraction": "flagship",
            },
        }
    )


class LLMServiceTests(unittest.TestCase):
    def test_generate_text_routes_to_configured_provider(self) -> None:
        provider = FakeProvider()
        service = LLMService(config(), providers={"azure": provider})

        response = service.generate_text(
            role="summarization",
            messages=[LLMMessage(role="user", content="Summarize this.")],
        )

        self.assertEqual(response.text, "hello")
        self.assertEqual(provider.requests[0].model, "nextsearch-fast")

    def test_generate_json_validates_provider_result(self) -> None:
        provider = FakeProvider()
        service = LLMService(config(), providers={"azure": provider})

        result = service.generate_json(
            role="graph_extraction",
            messages=[LLMMessage(role="user", content="Extract facts.")],
            response_model=Answer,
        )

        self.assertEqual(result, Answer(value="ok"))
        self.assertEqual(provider.requests[0].model, "nextsearch-flagship")

    def test_generate_json_rejects_invalid_provider_result(self) -> None:
        provider = FakeProvider()
        provider.json_result = {"missing": "value"}
        service = LLMService(config(), providers={"azure": provider})

        with self.assertRaises(LLMStructuredOutputError):
            service.generate_json(
                role="graph_extraction",
                messages=[LLMMessage(role="user", content="Extract facts.")],
                response_model=Answer,
            )

    def test_embed_routes_to_configured_provider(self) -> None:
        provider = FakeProvider()
        service = LLMService(config(), providers={"azure": provider})

        response = service.embed(
            role="document_embedding",
            texts=["Document text"],
        )

        self.assertEqual(response.embeddings, [[0.1, 0.2]])
        self.assertEqual(provider.requests[0].model, "nextsearch-embed")

    def test_embedding_accessors_return_configured_defaults(self) -> None:
        service = LLMService(config(), providers={"azure": FakeProvider()})

        self.assertEqual(service.embedding_provider_name(), "azure")
        self.assertEqual(service.embedding_model(), "nextsearch-embed")


if __name__ == "__main__":
    unittest.main()
