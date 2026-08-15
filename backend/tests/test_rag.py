import asyncio

from poc_kanini.core.config import Settings
from poc_kanini.models.documents import DocumentMetadata, DocumentStructure, ProcessedDocument, ProcessedPage
from poc_kanini.rag.models import Citation, DocumentChunk, RetrievedChunk
from poc_kanini.rag.retrieval import RetrievalService
from poc_kanini.rag.service import RagService


class FakeEmbeddings:
    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    async def embed_documents(self, chunks):
        self.document_calls += 1
        return [[float(index + 1)] for index, _ in enumerate(chunks)]

    async def embed_query(self, query):
        self.query_calls += 1
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


def test_repeated_retrieval_of_an_indexed_document_does_not_reembed_chunks():
    """Subsequent RAG questions embed only their query; stored chunk vectors are reused."""
    embeddings = FakeEmbeddings()
    service = RagService(Settings(), FakeStore([]), embeddings=embeddings)
    asyncio.run(service.index_document(document()))
    asyncio.run(service.retrieve_evidence("What is the leave allowance?", document_id="doc-1"))
    asyncio.run(service.retrieve_evidence("How many leave days are available?", document_id="doc-1"))

    assert embeddings.document_calls == 1
    assert embeddings.query_calls == 2


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


def test_retrieve_evidence_decoupled_from_llm():
    """retrieve_evidence must return chunks and citations without requiring Gemini API key or LLM generation."""
    service = RagService(Settings(gemini_api_key=None), FakeStore([chunk()]), embeddings=FakeEmbeddings())
    sources, citations = asyncio.run(service.retrieve_evidence("annual leave", document_id="doc-1"))
    assert len(sources) == 1
    assert sources[0].chunk.text == "The annual leave allowance is twenty days."
    assert len(citations) == 1
    assert citations[0].label == "policy.pdf — Page 2"


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


def test_rag_topic_extraction_fallback_without_gemini():
    """When Gemini is unavailable, simple topic extraction returns clean structured bullets."""
    from poc_kanini.graphs.specialists import synthesize_node
    from langchain_core.messages import HumanMessage
    from unittest.mock import patch

    curriculum_text = (
        "1. Text, Image & Audio Processing Foundations\n"
        "2. Embeddings, Representations & GenAI Basics\n"
        "3. Advanced Google GenAI Capabilities\n"
        "4. Intelligent Document Processing Systems\n"
        "5. Vector Databases & Retrieval Systems"
    )
    state = {
        "route": "rag",
        "messages": [HumanMessage(content="give me any 3 topics from the given curriculum")],
        "tool_results": [{
            "tool": "search_document_evidence",
            "query": "give me any 3 topics from the given curriculum",
            "result": {
                "evidence": [{"text": curriculum_text}],
                "citations": [{"filename": "Data Science Curriculum.pdf", "page_number": 1}],
            },
        }],
        "activities": [], "warnings": [], "reports": [], "actions": [], "step_count": 1,
    }

    with patch("poc_kanini.graphs.specialists.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = None
        content = asyncio.run(synthesize_node(state))["messages"][-1].content

    assert "Text, Image & Audio Processing Foundations" in content
    assert "Embeddings, Representations & GenAI Basics" in content
    assert "Advanced Google GenAI Capabilities" in content
    assert "[Data Science Curriculum.pdf — Page 1]" in content
    # Ensure it's not a raw chunk dump or generic 'unable to synthesize'
    assert content.count("- ") == 3


def test_rag_after_ml_conversation_does_not_leak_ml_history():
    """Stateful multi-turn conversation switching from ML to RAG must not leak ML context or reports."""
    from fastapi.testclient import TestClient
    from poc_kanini.main import app
    from unittest.mock import patch

    curriculum_text = (
        "1. Text, Image & Audio Processing Foundations\n"
        "2. Embeddings, Representations & GenAI Basics\n"
        "3. Advanced Google GenAI Capabilities\n"
    )

    with TestClient(app) as client, \
         patch("poc_kanini.rag.service.RagService.retrieve_evidence") as mock_retrieve:
        mock_retrieve.return_value = (
            [RetrievedChunk(chunk=DocumentChunk(chunk_id="c1", document_id="doc-curr", filename="curriculum.pdf", document_type="curriculum", page_number=1, chunk_index=0, text=curriculum_text), score=0.95, distance=0.05)],
            [Citation(document_id="doc-curr", filename="curriculum.pdf", page_number=1, label="curriculum.pdf — Page 1", chunk_id="c1", score=0.95)],
        )

        # Turn 1: ML training
        train_prompt = "Train a classification model using this dataset: feature1, feature2, churn. Use these records: (1,2,0), (2,3,0), (8,9,1), (9,10,0), (3,4,0), (7,8,1)."
        r1 = client.post("/api/chat", json={"messages": [{"role": "user", "content": train_prompt}]})
        assert r1.status_code == 200
        thread_id = r1.json()["thread_id"]
        appr_id = r1.json()["approval_id"]

        # Turn 2: Approve
        r2 = client.post("/api/chat/approval", json={"thread_id": thread_id, "decision": "approved", "approval_id": appr_id})
        assert r2.status_code == 200

        # Turn 3: Ask RAG question with attached curriculum document
        rag_prompt = "give me any 3 topics from the given curriculum"
        r3 = client.post("/api/chat", json={"thread_id": thread_id, "document_id": "doc-curr", "messages": [{"role": "user", "content": rag_prompt}]})
        assert r3.status_code == 200
        b3 = r3.json()

        ans = b3.get("message", {}).get("content", "")
        # Must contain curriculum topics
        assert "Text, Image & Audio Processing Foundations" in ans or "Embeddings" in ans
        # Must NOT contain ML model training leakage
        assert "Trained LogisticRegression" not in ans
        assert "Trained RandomForest" not in ans
        assert "Dataset contains" not in ans
        # Must NOT contain dataset_analysis reports
        reports = b3.get("reports", [])
        assert not any(rep.get("report_type") == "dataset_analysis" for rep in reports)


def test_rag_multiturn_topic_continuation_and_exclusion():
    """Multi-turn test for topic extraction:
    Turn 1: 'give me any 3 topics from the given curriculum' -> topics 1-3
    Turn 2: 'give me more topics' -> topics 4-6
    Turn 3: 'anything other than the three topics you already gave me' -> topics 7-9
    Turn 4: 'give me more topics' -> no additional topics message
    """
    from fastapi.testclient import TestClient
    from poc_kanini.main import app
    from unittest.mock import patch

    full_curriculum_text = (
        "1. Text, Image & Audio Processing Foundations\n"
        "2. Embeddings, Representations & GenAI Basics\n"
        "3. Advanced Google GenAI Capabilities\n"
        "4. Intelligent Document Processing Systems\n"
        "5. Vector Databases & Retrieval Systems\n"
        "6. Multimodal AI with Gemini\n"
        "7. Foundations of Autonomous Agents\n"
        "8. Agent Function Calling & Tooling\n"
        "9. Agent Memory & State Management\n"
    )

    with TestClient(app) as client, \
         patch("poc_kanini.rag.service.RagService.retrieve_evidence") as mock_retrieve:
        mock_retrieve.return_value = (
            [RetrievedChunk(chunk=DocumentChunk(chunk_id="c1", document_id="doc-curr", filename="curriculum.pdf", document_type="curriculum", page_number=1, chunk_index=0, text=full_curriculum_text), score=0.95, distance=0.05)],
            [Citation(document_id="doc-curr", filename="curriculum.pdf", page_number=1, label="curriculum.pdf — Page 1", chunk_id="c1", score=0.95)],
        )

        # Turn 1: Initial request for 3 topics
        r1 = client.post("/api/chat", json={"document_id": "doc-curr", "messages": [{"role": "user", "content": "give me any 3 topics from the given curriculum"}]})
        assert r1.status_code == 200
        b1 = r1.json()
        thread_id = b1["thread_id"]
        c1 = b1["message"]["content"]
        assert "Text, Image & Audio Processing Foundations" in c1
        assert "Embeddings, Representations & GenAI Basics" in c1
        assert "Advanced Google GenAI Capabilities" in c1
        assert "Intelligent Document Processing Systems" not in c1
        assert "[curriculum.pdf — Page 1]" in c1

        # Turn 2: Request more topics
        r2 = client.post("/api/chat", json={"thread_id": thread_id, "document_id": "doc-curr", "messages": [{"role": "user", "content": "give me more topics"}]})
        assert r2.status_code == 200
        b2 = r2.json()
        c2 = b2["message"]["content"]
        assert "Intelligent Document Processing Systems" in c2
        assert "Vector Databases & Retrieval Systems" in c2
        assert "Multimodal AI with Gemini" in c2
        # Must NOT repeat topics from Turn 1
        assert "Text, Image & Audio Processing Foundations" not in c2
        assert "Embeddings, Representations & GenAI Basics" not in c2
        assert "[curriculum.pdf — Page 1]" in c2

        # Turn 3: Request anything other than the topics given
        r3 = client.post("/api/chat", json={"thread_id": thread_id, "document_id": "doc-curr", "messages": [{"role": "user", "content": "anything other than the three topics you already gave me"}]})
        assert r3.status_code == 200
        b3 = r3.json()
        c3 = b3["message"]["content"]
        assert "Foundations of Autonomous Agents" in c3
        assert "Agent Function Calling & Tooling" in c3
        assert "Agent Memory & State Management" in c3
        # Must NOT repeat topics from Turn 1 or Turn 2
        assert "Text, Image & Audio Processing Foundations" not in c3
        assert "Intelligent Document Processing Systems" not in c3
        assert "[curriculum.pdf — Page 1]" in c3

        # Turn 4: Exhausted topics request
        r4 = client.post("/api/chat", json={"thread_id": thread_id, "document_id": "doc-curr", "messages": [{"role": "user", "content": "give me more topics"}]})
        assert r4.status_code == 200
        b4 = r4.json()
        c4 = b4["message"]["content"]
        assert "no additional top-level curriculum topics" in c4.lower() or "no additional" in c4.lower()


def test_rag_specific_section_and_referential_followup():
    """Test specific section extraction and referential follow-up:
    Turn 1: 'What does the curriculum say about Vector Databases & Retrieval Systems?'
            -> Scoped strictly to Vector Databases section and its subsections
            -> Stops before Section 6 (Multimodal AI with Gemini)
            -> Formatted as: **Vector Databases & Retrieval Systems** covers:
    Turn 2: 'What are the specific technologies or methods mentioned in that section?'
            -> Resolves 'that section' to Vector Databases & Retrieval Systems
            -> Returns specific technologies (FAISS, ChromaDB, Pinecone, Hybrid Search, Cross-Encoder)
            -> Does NOT return generic top-level topics 1-3
    """
    from fastapi.testclient import TestClient
    from poc_kanini.main import app
    from unittest.mock import patch

    curriculum_evidence = (
        "4. Intelligent Document Processing Systems\n"
        "4.1 OCR Extraction Pipelines\n"
        "5. Vector Databases & Retrieval Systems\n"
        "5.1 Semantic Dense Representation Models\n"
        "5.2 Vector Similarity Metrics Mechanics\n"
        "5.3 FAISS Local Indexing Implementation\n"
        "5.4 ChromaDB local Cloud Setup\n"
        "5.5 Pinecone Enterprise Index Cluster\n"
        "5.6 Hybrid Search Keyword Merging\n"
        "5.7 Cross-Encoder Reranking Execution Models\n"
        "6. Multimodal AI with Gemini\n"
        "6.1 Vision Models Integration\n"
    )

    with TestClient(app) as client, \
         patch("poc_kanini.rag.service.RagService.retrieve_evidence") as mock_retrieve:
        mock_retrieve.return_value = (
            [RetrievedChunk(chunk=DocumentChunk(chunk_id="c5", document_id="doc-curr", filename="curriculum.pdf", document_type="curriculum", page_number=2, chunk_index=4, text=curriculum_evidence), score=0.95, distance=0.05)],
            [Citation(document_id="doc-curr", filename="curriculum.pdf", page_number=2, label="curriculum.pdf — Page 2", chunk_id="c5", score=0.95)],
        )

        # Turn 1: Specific section query
        r1 = client.post(
            "/api/chat",
            json={
                "document_id": "doc-curr",
                "messages": [{"role": "user", "content": "What does the curriculum say about Vector Databases & Retrieval Systems?"}],
            },
        )
        assert r1.status_code == 200
        b1 = r1.json()
        thread_id = b1["thread_id"]
        c1 = b1["message"]["content"]

        # Must be scoped strictly to Section 5
        assert "Vector Databases & Retrieval Systems" in c1
        assert "FAISS Local Indexing Implementation" in c1
        assert "ChromaDB local Cloud Setup" in c1
        assert "Pinecone Enterprise Index Cluster" in c1
        assert "Hybrid Search Keyword Merging" in c1
        assert "Cross-Encoder Reranking Execution Models" in c1

        # Must NOT leak Section 4 or Section 6
        assert "OCR Extraction Pipelines" not in c1
        assert "Vision Models Integration" not in c1
        assert "[curriculum.pdf — Page 2]" in c1

        # Turn 2: Referential follow-up
        r2 = client.post(
            "/api/chat",
            json={
                "thread_id": thread_id,
                "document_id": "doc-curr",
                "messages": [{"role": "user", "content": "What are the specific technologies or methods mentioned in that section?"}],
            },
        )
        assert r2.status_code == 200
        b2 = r2.json()
        c2 = b2["message"]["content"]

        # Must answer using the Vector Databases section technologies
        assert "Vector Databases & Retrieval Systems" in c2 or "FAISS" in c2
        assert "FAISS Local Indexing Implementation" in c2
        assert "ChromaDB local Cloud Setup" in c2
        assert "Pinecone Enterprise Index Cluster" in c2

        # Must NOT return generic top-level topics 1-3
        assert "Text, Image & Audio Processing Foundations" not in c2
        assert "Embeddings, Representations & GenAI Basics" not in c2
        assert "[curriculum.pdf — Page 2]" in c2


def test_rag_synthetic_controlled_topic_continuation_and_chunk_reordering():
    """Verify that continuation is genuinely grounded in retrieved document evidence,
    correctly extracts ordered topic sets, excludes previously presented topics,
    handles chunk reordering/partial chunks, and exhausts gracefully."""
    from fastapi.testclient import TestClient
    from poc_kanini.main import app
    from unittest.mock import patch

    synthetic_chunk_1 = (
        "1. Alpha Topic\n"
        "2. Beta Topic\n"
        "3. Gamma Topic\n"
    )
    synthetic_chunk_2 = (
        "4. Delta Topic\n"
        "5. Epsilon Topic\n"
        "6. Zeta Topic\n"
    )

    with TestClient(app) as client:
        # Turn 1: Normal document with both chunks available
        with patch("poc_kanini.rag.service.RagService.retrieve_evidence") as mock_retrieve:
            mock_retrieve.return_value = (
                [
                    RetrievedChunk(chunk=DocumentChunk(chunk_id="sc1", document_id="doc-syn", filename="syllabus.pdf", document_type="syllabus", page_number=1, chunk_index=0, text=synthetic_chunk_1), score=0.95, distance=0.05),
                    RetrievedChunk(chunk=DocumentChunk(chunk_id="sc2", document_id="doc-syn", filename="syllabus.pdf", document_type="syllabus", page_number=2, chunk_index=1, text=synthetic_chunk_2), score=0.90, distance=0.10),
                ],
                [
                    Citation(document_id="doc-syn", filename="syllabus.pdf", page_number=1, label="syllabus.pdf — Page 1", chunk_id="sc1", score=0.95),
                    Citation(document_id="doc-syn", filename="syllabus.pdf", page_number=2, label="syllabus.pdf — Page 2", chunk_id="sc2", score=0.90),
                ],
            )

            r1 = client.post("/api/chat", json={"document_id": "doc-syn", "messages": [{"role": "user", "content": "give me 3 topics"}]})
            assert r1.status_code == 200
            b1 = r1.json()
            thread_id = b1["thread_id"]
            c1 = b1["message"]["content"]
            assert "Alpha Topic" in c1
            assert "Beta Topic" in c1
            assert "Gamma Topic" in c1
            assert "Delta Topic" not in c1
            assert "[syllabus.pdf — Page 1]" in c1 or "[syllabus.pdf — Page 2]" in c1

        # Turn 2: Chunks returned in REVERSED order (Chunk 2 then Chunk 1)
        with patch("poc_kanini.rag.service.RagService.retrieve_evidence") as mock_retrieve:
            mock_retrieve.return_value = (
                [
                    RetrievedChunk(chunk=DocumentChunk(chunk_id="sc2", document_id="doc-syn", filename="syllabus.pdf", document_type="syllabus", page_number=2, chunk_index=1, text=synthetic_chunk_2), score=0.95, distance=0.05),
                    RetrievedChunk(chunk=DocumentChunk(chunk_id="sc1", document_id="doc-syn", filename="syllabus.pdf", document_type="syllabus", page_number=1, chunk_index=0, text=synthetic_chunk_1), score=0.90, distance=0.10),
                ],
                [
                    Citation(document_id="doc-syn", filename="syllabus.pdf", page_number=2, label="syllabus.pdf — Page 2", chunk_id="sc2", score=0.95),
                    Citation(document_id="doc-syn", filename="syllabus.pdf", page_number=1, label="syllabus.pdf — Page 1", chunk_id="sc1", score=0.90),
                ],
            )

            r2 = client.post("/api/chat", json={"thread_id": thread_id, "document_id": "doc-syn", "messages": [{"role": "user", "content": "give me 3 more topics"}]})
            assert r2.status_code == 200
            b2 = r2.json()
            c2 = b2["message"]["content"]
            # Must return Delta, Epsilon, Zeta
            assert "Delta Topic" in c2
            assert "Epsilon Topic" in c2
            assert "Zeta Topic" in c2
            # Must NOT repeat Alpha, Beta, Gamma even if Chunk 1 was also retrieved
            assert "Alpha Topic" not in c2
            assert "Beta Topic" not in c2
            assert "Gamma Topic" not in c2

        # Turn 3: All 6 topics have been presented -> honest exhaustion
        with patch("poc_kanini.rag.service.RagService.retrieve_evidence") as mock_retrieve:
            mock_retrieve.return_value = (
                [
                    RetrievedChunk(chunk=DocumentChunk(chunk_id="sc1", document_id="doc-syn", filename="syllabus.pdf", document_type="syllabus", page_number=1, chunk_index=0, text=synthetic_chunk_1), score=0.95, distance=0.05),
                    RetrievedChunk(chunk=DocumentChunk(chunk_id="sc2", document_id="doc-syn", filename="syllabus.pdf", document_type="syllabus", page_number=2, chunk_index=1, text=synthetic_chunk_2), score=0.90, distance=0.10),
                ],
                [
                    Citation(document_id="doc-syn", filename="syllabus.pdf", page_number=1, label="syllabus.pdf — Page 1", chunk_id="sc1", score=0.95),
                ],
            )

            r3 = client.post("/api/chat", json={"thread_id": thread_id, "document_id": "doc-syn", "messages": [{"role": "user", "content": "give me more"}]})
            assert r3.status_code == 200
            b3 = r3.json()
            c3 = b3["message"]["content"]
            assert "no additional top-level curriculum topics" in c3.lower() or "no additional" in c3.lower()




