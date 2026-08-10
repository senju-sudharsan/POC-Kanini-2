"""Safe PDF ingestion and structured semantic parsing, deliberately without retrieval or RAG."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from poc_kanini.documents.classification import classify_document
from poc_kanini.documents.layout import analyze_layout
from poc_kanini.documents.ocr import OcrEngine, TesseractOcrEngine
from poc_kanini.models.documents import DocumentMetadata, ProcessedDocument, ProcessedPage
from poc_kanini.nlp.processor import NlpProcessor


class DocumentValidationError(ValueError):
    """Raised when an uploaded document is unsafe or outside Phase 2 support."""


class DocumentProcessor:
    """Convert a safe PDF upload into provenance-preserving page representations."""

    supported_content_types = {"application/pdf", "application/x-pdf"}

    def __init__(self, nlp: NlpProcessor | None = None, ocr_engine: OcrEngine | None = None, max_file_size_bytes: int = 20 * 1024 * 1024, minimum_text_characters: int = 20) -> None:
        self._nlp = nlp or NlpProcessor()
        self._ocr_engine = ocr_engine or TesseractOcrEngine()
        self._max_file_size_bytes = max_file_size_bytes
        self._minimum_text_characters = minimum_text_characters

    def process(self, content: bytes, filename: str, content_type: str | None = None) -> ProcessedDocument:
        """Validate and parse a PDF, invoking OCR only where native extraction is poor."""

        safe_filename = Path(filename).name
        self._validate(content, safe_filename, content_type)
        try:
            reader = PdfReader(BytesIO(content))
        except Exception as error:
            raise DocumentValidationError("The uploaded file is not a readable PDF.") from error
        if not reader.pages:
            raise DocumentValidationError("The uploaded PDF contains no pages.")

        notes: list[str] = []
        pages: list[ProcessedPage] = []
        for number, page in enumerate(reader.pages, start=1):
            extracted = (page.extract_text() or "").strip()
            method = "pdf_text"
            ocr_status = "not_required"
            if len(extracted) < self._minimum_text_characters:
                extracted, method, ocr_status, note = self._ocr_or_empty(content, number, extracted)
                if note:
                    notes.append(note)
            normalized = self._nlp.normalize(extracted) if extracted else ""
            pages.append(ProcessedPage(
                page_number=number,
                text=extracted,
                normalized_text=normalized,
                semantic_terms=self._nlp.semantic_clean(extracted) if extracted else [],
                extraction_method=method,
                ocr_status=ocr_status,
                structure=analyze_layout(extracted),
            ))

        all_text = "\n".join(page.text for page in pages)
        metadata = DocumentMetadata(
            document_id=sha256(content).hexdigest(),
            filename=safe_filename,
            content_type=content_type,
            file_size_bytes=len(content),
            page_count=len(pages),
            pdf_metadata={str(key).lstrip("/"): str(value) for key, value in (reader.metadata or {}).items() if value is not None},
        )
        return ProcessedDocument(metadata=metadata, document_type=classify_document(all_text), pages=pages, processing_notes=notes)

    def _validate(self, content: bytes, filename: str, content_type: str | None) -> None:
        if not content:
            raise DocumentValidationError("The uploaded file is empty.")
        if len(content) > self._max_file_size_bytes:
            raise DocumentValidationError("The uploaded file exceeds the 20 MiB limit.")
        if Path(filename).suffix.casefold() != ".pdf":
            raise DocumentValidationError("Only PDF files are supported in Phase 2.")
        if content_type and content_type.casefold() not in self.supported_content_types:
            raise DocumentValidationError("The uploaded content type is not a PDF.")
        if not content.lstrip().startswith(b"%PDF-"):
            raise DocumentValidationError("The uploaded file does not have a valid PDF signature.")

    def _ocr_or_empty(self, content: bytes, page_number: int, extracted: str) -> tuple[str, str, str, str | None]:
        if not self._ocr_engine.is_available():
            return extracted, "empty", "unavailable", f"Page {page_number}: native extraction was poor and local OCR is unavailable."
        try:
            ocr_text = self._ocr_engine.extract_page(content, page_number).strip()
        except Exception:
            return extracted, "empty", "failed", f"Page {page_number}: OCR fallback failed."
        if not ocr_text:
            return extracted, "empty", "completed", f"Page {page_number}: OCR returned no text."
        return ocr_text, "ocr", "completed", f"Page {page_number}: OCR fallback used."
