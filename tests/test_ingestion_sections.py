import unittest
from pathlib import Path

from nextsearch.ingestion.markdown.sections import split_markdown_into_sections
from nextsearch.ingestion.models import MarkdownDocument


class MarkdownSectionTests(unittest.TestCase):
    def test_split_markdown_by_headings_and_preserves_page_ranges(self) -> None:
        document = _document(
            "<!-- page: 1 -->\n"
            "# Overview\n"
            "Project overview.\n\n"
            "<!-- page: 2 -->\n"
            "## Dependencies\n"
            "Vendor dependencies.\n"
        )

        sections = split_markdown_into_sections(document)

        self.assertEqual([section.heading for section in sections], ["Overview", "Dependencies"])
        self.assertEqual(sections[0].heading_path, ("Overview",))
        self.assertEqual(sections[1].heading_path, ("Overview", "Dependencies"))
        self.assertEqual((sections[0].page_start, sections[0].page_end), (1, 1))
        self.assertEqual((sections[1].page_start, sections[1].page_end), (2, 2))
        self.assertIn("<!-- page: 2 -->", sections[1].text)

    def test_split_markdown_handles_text_before_first_heading(self) -> None:
        document = _document(
            "<!-- page: 1 -->\n"
            "Preface text before headings.\n\n"
            "# Overview\n"
            "Overview text.\n"
        )

        sections = split_markdown_into_sections(document)

        self.assertEqual([section.heading for section in sections], ["Untitled", "Overview"])
        self.assertEqual(sections[0].heading_path, ("Untitled",))
        self.assertEqual((sections[0].page_start, sections[0].page_end), (1, 1))
        self.assertEqual((sections[1].page_start, sections[1].page_end), (1, 1))

    def test_split_markdown_splits_oversized_sections_by_paragraph(self) -> None:
        document = _document(
            "<!-- page: 1 -->\n"
            "# Large Section\n\n"
            "Alpha\n\n"
            "Beta\n\n"
            "Gamma\n"
        )

        sections = split_markdown_into_sections(document, max_section_chars=35)

        self.assertEqual(
            [section.id for section in sections],
            ["section-0001-part-0001", "section-0001-part-0002"],
        )
        self.assertEqual([section.heading for section in sections], ["Large Section", "Large Section"])
        self.assertEqual(
            [(section.page_start, section.page_end) for section in sections],
            [(1, 1), (1, 1)],
        )


def _document(markdown: str) -> MarkdownDocument:
    return MarkdownDocument(
        markdown=markdown,
        source_path=Path("sample.pdf"),
        page_count=2,
        batches=(),
    )


if __name__ == "__main__":
    unittest.main()
