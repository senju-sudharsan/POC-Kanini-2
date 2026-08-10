import asyncio

from poc_kanini.core.config import Settings
from poc_kanini.models.documents import DocumentMetadata, DocumentStructure, ProcessedDocument, ProcessedPage
from poc_kanini.rag.models import DocumentChunk, RetrievedChunk
from poc_kanini.rag.retrieval import RetrievalService
from poc_kanini.rag.service import RagService


class FakeEmbeddings:
    async def embed_documents(self, chunks):
        return [[float(index + 1)] for index, _ in enumerate(chunks)]

    async def embed_query(self, query):
        return [1.0]


class FakeStore:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.deleted = []
        self.upserted = []

    def upsert(self, chunks, embeddings):
        self.upserted = list(zip(chunks, embeddings))
        self.chunks = list(chunks)

    def delete_document(self, document_id):
        self.deleted.append(document_id)
        self.chunks = [chunk for chunk in self.chunks if chunk.document_id != document_id]

    def similarity_search(self, embedding, limit, document_id=None):
        matches = [chunk for chunk in self.chunks if document_id is None or chunk.document_id == document_id]
        return [RetrievedChunk(chunk=chunk, distance=0.2, score=0.8) for chunk in matches[:limit]]

    def all_chunks(self, document_id=None):
        return [chunk for chunk in self.chunks if document_id is None or chunk.document_id == document_id]


def chunk(chunk_id="one", text="The annual leave allowance is twenty days.", document_id="doc-1"):
    return DocumentChunk(chunk_id=chunk_id, document_id=document_id, filename="policy.pdf", document_type="policy", page_number=2, chunk_index=0, text=text, metadata={"document_id": document_id, "filename": "policy.pdf", "document_type": "policy", "page_number": "2", "chunk_index": "0"})


def document():
    return ProcessedDocument(metadata=DocumentMetadata(document_id="doc-1", filename="policy.pdf", file_size_bytes=1, page_count=1), document_type="policy", pages=[ProcessedPage(page_number=1, text="Annual leave is twenty days.", normalized_text="annual leave is twenty days", semantic_terms=["annual", "leave"], extraction_method="pdf_text", ocr_status="not_required", structure=DocumentStructure(paragraphs=["Annual leave is twenty days."]))])


def test_hybrid_retrieval_preserves_provenance_and_keyword_score():
    result = asyncio.run(RetrievalService(FakeEmbeddings(), FakeStore([chunk()])).retrieve("annual leave", top_k=1, document_id="doc-1"))
    assert len(result) == 1
    assert result[0].chunk.filename == "policy.pdf"
    assert result[0].chunk.page_number == 2
    assert result[0].keyword_score > 0


def test_retrieval_filters_to_requested_document():
    store = FakeStore([chunk("one", document_id="doc-1"), chunk("two", document_id="doc-2")])
    result = asyncio.run(RetrievalService(FakeEmbeddings(), store).retrieve("annual leave", document_id="doc-1"))
    assert [item.chunk.document_id for item in result] == ["doc-1"]


def test_indexing_replaces_prior_document_chunks():
    store = FakeStore([chunk("old", document_id="doc-1")])
    service = RagService(Settings(), store, embeddings=FakeEmbeddings())
    count = asyncio.run(service.index_document(document()))
    assert count == 1
    assert store.deleted == ["doc-1"]
    assert store.upserted[0][0].document_id == "doc-1"


def test_insufficient_evidence_returns_no_fabricated_citations():
    service = RagService(Settings(), FakeStore([]), embeddings=FakeEmbeddings())
    result = asyncio.run(service.answer("What is the leave policy?"))
    assert "could not find enough" in result.answer
    assert result.citations == []


def test_citation_is_derived_from_retrieved_chunk():
    from poc_kanini.rag.service import _citation
    citation = _citation(RetrievedChunk(chunk=chunk()))
    assert citation.label == "policy.pdf — Page 2"
    assert citation.chunk_id == "one"


def test_grounded_response_uses_mocked_gemini_and_returns_retrieved_citation(monkeypatch):
    class FakeModels:
        async def generate_content(self, **kwargs):
            assert "Retrieved evidence" in kwargs["contents"]
            assert "policy.pdf — Page 2" in kwargs["contents"]
            return type("Response", (), {"text": "The allowance is twenty days. [policy.pdf — Page 2]"})()

    class FakeClient:
        def __init__(self, **kwargs):
            self.aio = type("Aio", (), {"models": FakeModels()})()

    import poc_kanini.rag.service as rag_module
    monkeypatch.setattr(rag_module.genai, "Client", FakeClient)
    settings = Settings(gemini_api_key="test-key")
    result = asyncio.run(RagService(settings, FakeStore([chunk()]), embeddings=FakeEmbeddings()).answer("What is the allowance?"))
    assert result.answer.startswith("The allowance")
    assert [citation.label for citation in result.citations] == ["policy.pdf — Page 2"]
