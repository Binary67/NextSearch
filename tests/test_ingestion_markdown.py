import tempfile
import unittest
from pathlib import Path
from typing import Any

from nextsearch.ingestion.errors import MarkdownExtractionError
from nextsearch.ingestion.markdown.llm_extractor import (
    extract_markdown,
    split_pages_for_markdown,
)
from nextsearch.ingestion.models import PageContent
from nextsearch.ingestion.pipeline import extract_pdf_to_markdown
from nextsearch.llm.types import LLMMessage, LLMResponse
from tests.pdf_fixture import build_text_pdf


class FakeLLM:
    def __init__(self, output: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.output = output

    def generate_text(
        self,
        *,
        role: str,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        text = self.output
        if text is None:
            text = "\n".join(
                f"<!-- page: {page_number} -->\n# Page {page_number}"
                for page_number in _page_numbers(messages[-1].content)
            )
        return LLMResponse(
            text=text,
            provider="fake",
            model="fake-model",
            usage={"output_tokens": 4},
        )


class MarkdownExtractionTests(unittest.TestCase):
    def test_split_pages_batches_up_to_five_pages(self) -> None:
        pages = [PageContent(page_number=page, text="text") for page in range(1, 7)]

        batches = split_pages_for_markdown(pages)

        self.assertEqual(
            [[page.page_number for page in batch] for batch in batches],
            [[1, 2, 3, 4, 5], [6]],
        )

    def test_split_pages_keeps_small_document_in_one_batch(self) -> None:
        pages = [PageContent(page_number=page, text="text") for page in range(1, 4)]

        batches = split_pages_for_markdown(pages)

        self.assertEqual(len(batches), 1)
        self.assertEqual([page.page_number for page in batches[0]], [1, 2, 3])

    def test_split_pages_splits_text_heavy_page_by_paragraphs(self) -> None:
        pages = [PageContent(page_number=1, text="a" * 8 + "\n\n" + "b" * 8)]

        batches = split_pages_for_markdown(
            pages,
            max_pages_per_batch=5,
            max_batch_chars=10,
        )

        self.assertEqual(len(batches), 2)
        self.assertEqual([batch[0].page_number for batch in batches], [1, 1])
        self.assertEqual([batch[0].text for batch in batches], ["a" * 8, "b" * 8])

    def test_extract_markdown_routes_to_markdown_extraction_role(self) -> None:
        llm = FakeLLM()
        pages = [
            PageContent(page_number=1, text="Title"),
            PageContent(page_number=2, text="Body"),
        ]

        document = extract_markdown(
            pages=pages,
            llm=llm,  # type: ignore[arg-type]
            source_path=Path("sample.pdf"),
        )

        self.assertEqual(llm.calls[0]["role"], "markdown_extraction")
        self.assertEqual(llm.calls[0]["temperature"], 0)
        self.assertIn("<!-- page: 1 -->", document.markdown)
        self.assertIn("<!-- page: 2 -->", document.markdown)
        self.assertEqual(document.batches[0].page_start, 1)
        self.assertEqual(document.batches[0].page_end, 2)
        self.assertEqual(document.batches[0].usage, {"output_tokens": 4})

    def test_extract_markdown_rejects_missing_page_anchor(self) -> None:
        llm = FakeLLM(output="# Missing anchor")

        with self.assertRaises(MarkdownExtractionError):
            extract_markdown(
                pages=[PageContent(page_number=1, text="Title")],
                llm=llm,  # type: ignore[arg-type]
                source_path=Path("sample.pdf"),
            )

    def test_pipeline_writes_markdown_artifacts(self) -> None:
        llm = FakeLLM()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "sample.pdf"
            output_dir = root / "out"
            pdf_path.write_bytes(build_text_pdf(["Title\nBody"]))

            document = extract_pdf_to_markdown(
                pdf_path,
                llm,  # type: ignore[arg-type]
                output_dir=output_dir,
            )

            self.assertEqual(document.page_count, 1)
            self.assertTrue((output_dir / "document.md").exists())
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "batches" / "batch-0001.output.md").exists())


def _page_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for line in text.splitlines():
        if line.startswith("<!-- page: ") and line.endswith(" -->"):
            numbers.append(int(line.removeprefix("<!-- page: ").removesuffix(" -->")))
    return numbers


if __name__ == "__main__":
    unittest.main()
