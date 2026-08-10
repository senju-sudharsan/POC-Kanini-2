"""Optional local OCR adapter for image-based PDF pages."""

from typing import Protocol
import shutil


class OcrEngine(Protocol):
    """A replaceable page OCR contract, easy to fake in deterministic tests."""

    def is_available(self) -> bool: ...

    def extract_page(self, pdf_bytes: bytes, page_number: int) -> str: ...


class TesseractOcrEngine:
    """Render a PDF page locally and OCR it with a locally installed Tesseract binary."""

    def is_available(self) -> bool:
        return shutil.which("tesseract") is not None

    def extract_page(self, pdf_bytes: bytes, page_number: int) -> str:
        if not self.is_available():
            raise RuntimeError("Tesseract is not installed or is not on PATH.")
        import pypdfium2 as pdfium
        import pytesseract

        document = pdfium.PdfDocument(pdf_bytes)
        page = document[page_number - 1]
        image = page.render(scale=2).to_pil()
        return pytesseract.image_to_string(image)
