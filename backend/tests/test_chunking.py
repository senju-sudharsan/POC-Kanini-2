from poc_kanini.models.documents import (
    DocumentMetadata,
    DocumentStructure,
    ProcessedDocument,
    ProcessedPage,
)
from poc_kanini.rag.chunking import DocumentChunker


def make_document() -> ProcessedDocument:
    return ProcessedDocument(
        metadata=DocumentMetadata(
            document_id="document-123",
            filename="company-policy.pdf",
            content_type="application/pdf",
            file_size_bytes=1000,
            page_count=2,
        ),
        document_type="policy",
        pages=[
            ProcessedPage(
                page_number=1,
                text="Leave policy. Employees receive annual leave.",
                normalized_text="leave policy employees receive annual leave",
                semantic_terms=["leave", "policy", "employees", "annual"],
                extraction_method="pdf_text",
                ocr_status="not_required",
                structure=DocumentStructure(
                    headings=["Leave Policy"],
                    paragraphs=[
                        "Employees receive annual leave according to company policy."
                    ],
                ),
            ),
            ProcessedPage(
                page_number=2,
                text="Sick leave requires notification.",
                normalized_text="sick leave requires notification",
                semantic_terms=["sick", "leave", "notification"],
                extraction_method="pdf_text",
                ocr_status="not_required",
                structure=DocumentStructure(
                    headings=["Sick Leave"],
                    paragraphs=[
                        "Sick leave requires notification to the manager."
                    ],
                ),
            ),
        ],
    )


def test_chunking_preserves_page_provenance():
    chunks = DocumentChunker().chunk(make_document())

    assert chunks
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(chunk.document_id == "document-123" for chunk in chunks)
    assert all(chunk.filename == "company-policy.pdf" for chunk in chunks)


def test_chunk_ids_are_deterministic():
    document = make_document()
    chunker = DocumentChunker()

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_chunk_indexes_are_sequential():
    chunks = DocumentChunker().chunk(make_document())

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_large_content_is_split():
    long_text = "This is an important policy statement. " * 100

    document = make_document()
    document.pages[0].structure = DocumentStructure(
        headings=["Large Policy"],
        paragraphs=[long_text],
    )
    document.pages[0].text = long_text
    document.pages[0].normalized_text = long_text

    chunks = DocumentChunker(
        target_characters=500,
        max_characters=700,
        overlap_characters=100,
    ).chunk(document)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 700 for chunk in chunks)


def test_fallback_text_is_used_when_layout_is_empty():
    document = make_document()
    document.pages[0].structure = DocumentStructure()
    document.pages[0].text = "Fallback paragraph content."
    document.pages[0].normalized_text = "Fallback paragraph content."

    chunks = DocumentChunker().chunk(document)

    assert any("Fallback paragraph content." in chunk.text for chunk in chunks)


def test_metadata_contains_citation_fields():
    chunk = DocumentChunker().chunk(make_document())[0]

    assert chunk.metadata["filename"] == "company-policy.pdf"
    assert chunk.metadata["page_number"] == str(chunk.page_number)
    assert chunk.metadata["document_id"] == "document-123"