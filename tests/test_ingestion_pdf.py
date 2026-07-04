import tempfile
import unittest
from pathlib import Path

from nextsearch.ingestion.errors import PDFExtractionError
from nextsearch.ingestion.sources.pdf import extract_pdf_pages
from tests.pdf_fixture import build_text_pdf


class PDFExtractionTests(unittest.TestCase):
    def test_extract_pdf_pages_returns_page_numbered_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample.pdf"
            pdf_path.write_bytes(
                build_text_pdf(["First Page\nHello world", "Second Page\nMore text"])
            )

            pages = extract_pdf_pages(pdf_path)

        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0].page_number, 1)
        self.assertEqual(pages[0].text, "First Page\nHello world")
        self.assertEqual(pages[1].page_number, 2)
        self.assertEqual(pages[1].text, "Second Page\nMore text")

    def test_extract_pdf_pages_rejects_missing_file(self) -> None:
        with self.assertRaises(PDFExtractionError):
            extract_pdf_pages(Path("missing.pdf"))

    def test_extract_pdf_pages_rejects_pages_without_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "blank.pdf"
            pdf_path.write_bytes(build_text_pdf([""]))

            with self.assertRaises(PDFExtractionError):
                extract_pdf_pages(pdf_path)


if __name__ == "__main__":
    unittest.main()
