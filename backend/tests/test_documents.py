from io import BytesIO

import pytest
from reportlab.pdfgen import canvas

from poc_kanini.documents.processor import DocumentProcessor, DocumentValidationError


def pdf_bytes(*pages: str) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    for page in pages:
        if page:
            text = pdf.beginText(72, 750)
            for line in page.split("\n"):
                text.textLine(line)
            pdf.drawText(text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


class FakeOcr:
    def __init__(self, available: bool = True, text: str = "Scanned policy text") -> None:
        self.available = available
        self.text = text
        self.calls: list[int] = []

    def is_available(self) -> bool:
        return self.available

    def extract_page(self, content: bytes, page_number: int) -> str:
        self.calls.append(page_number)
        return self.text


def test_ingests_pdf_extracts_pages_and_preserves_metadata() -> None:
    content = pdf_bytes("POLICY\n- Follow the controls\nOwner   Status", "Second page report text")
    document = DocumentProcessor().process(content, "../../Enterprise Policy.pdf", "application/pdf")
    assert document.metadata.filename == "Enterprise Policy.pdf"
    assert document.metadata.document_id
    assert document.metadata.page_count == 2
    assert [page.page_number for page in document.pages] == [1, 2]
    assert "follow the controls" in document.pages[0].normalized_text
    assert document.pages[0].structure.headings == ["POLICY"]
    assert document.pages[0].structure.list_items == ["- Follow the controls"]
    assert document.document_type == "policy"


def test_rejects_invalid_type_and_signature() -> None:
    processor = DocumentProcessor()
    with pytest.raises(DocumentValidationError, match="Only PDF"):
        processor.process(b"plain text", "notes.txt", "text/plain")
    with pytest.raises(DocumentValidationError, match="signature"):
        processor.process(b"plain text", "notes.pdf", "application/pdf")


def test_rejects_oversized_upload() -> None:
    with pytest.raises(DocumentValidationError, match="exceeds"):
        DocumentProcessor(max_file_size_bytes=4).process(b"%PDF-1.7", "large.pdf", "application/pdf")


def test_empty_page_uses_ocr_fallback_when_available() -> None:
    ocr = FakeOcr(text="Employee handbook content")
    document = DocumentProcessor(ocr_engine=ocr).process(pdf_bytes(""), "scan.pdf", "application/pdf")
    page = document.pages[0]
    assert ocr.calls == [1]
    assert page.extraction_method == "ocr"
    assert page.ocr_status == "completed"
    assert page.text == "Employee handbook content"
    assert document.document_type == "handbook"


def test_poor_extraction_records_unavailable_ocr_without_fabricating_text() -> None:
    document = DocumentProcessor(ocr_engine=FakeOcr(available=False)).process(pdf_bytes(""), "scan.pdf", "application/pdf")
    page = document.pages[0]
    assert page.text == ""
    assert page.extraction_method == "empty"
    assert page.ocr_status == "unavailable"
    assert document.processing_notes == ["Page 1: native extraction was poor and local OCR is unavailable."]


def test_structured_output_is_deterministic_for_same_content() -> None:
    content = pdf_bytes("GUIDELINE\nRecommended best practice")
    processor = DocumentProcessor()
    first = processor.process(content, "guide.pdf", "application/pdf")
    second = processor.process(content, "guide.pdf", "application/pdf")
    assert first.metadata.document_id == second.metadata.document_id
    assert first.document_type == "guideline"
    assert first.pages[0].semantic_terms
