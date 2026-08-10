"""Structured, page-level document representations for later chunking and citations."""

from typing import Literal

from pydantic import BaseModel, Field


DocumentCategory = Literal["policy", "report", "handbook", "guideline", "other"]
ExtractionMethod = Literal["pdf_text", "ocr", "empty"]
OcrStatus = Literal["not_required", "completed", "unavailable", "failed"]


class DocumentStructure(BaseModel):
    """Useful, lightweight layout signals extracted from one page."""

    headings: list[str] = Field(default_factory=list)
    paragraphs: list[str] = Field(default_factory=list)
    list_items: list[str] = Field(default_factory=list)
    table_lines: list[str] = Field(default_factory=list)


class ProcessedPage(BaseModel):
    """A provenance-preserving page ready for later chunking and citation work."""

    page_number: int = Field(ge=1)
    text: str
    normalized_text: str
    semantic_terms: list[str]
    extraction_method: ExtractionMethod
    ocr_status: OcrStatus
    structure: DocumentStructure


class DocumentMetadata(BaseModel):
    """Safe source and processing metadata for an ingested document."""

    document_id: str
    filename: str
    content_type: str | None = None
    file_size_bytes: int
    page_count: int
    pdf_metadata: dict[str, str] = Field(default_factory=dict)
    processing_version: str = "phase-2"


class ProcessedDocument(BaseModel):
    """Complete Phase 2 document-processing output with page-level provenance."""

    metadata: DocumentMetadata
    document_type: DocumentCategory
    pages: list[ProcessedPage]
    processing_notes: list[str] = Field(default_factory=list)
