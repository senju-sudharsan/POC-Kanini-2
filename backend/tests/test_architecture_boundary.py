"""Comprehensive Architecture & State-Boundary Multi-Turn Tests.

Verifies the systematic separation between long-lived conversational memory
and current-turn execution state (tool results, citations, reports, warnings).
All tests are deterministic and mock external providers to consume 0 live quota.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from poc_kanini.main import app
from poc_kanini.rag.models import Citation, DocumentChunk, RetrievedChunk


# ──────────────────────────────────────────────────────────────────────────────
# Scenario A: ML → RAG State Isolation
# ──────────────────────────────────────────────────────────────────────────────

def test_scenario_a_ml_to_rag_isolation():
    """Turn 1: ML training (HITL approved)
    Turn 2: Unrelated document question
    Expected: Turn 2 contains NO ML reports, NO ML metrics, and NO training warnings.
    """
    curriculum_evidence = (
        "5. Vector Databases & Retrieval Systems\n"
        "5.1 Semantic Dense Representation Models\n"
        "5.2 Vector Similarity Metrics Mechanics\n"
    )

    with TestClient(app) as client, \
         patch("poc_kanini.rag.service.RagService.retrieve_evidence") as mock_retrieve:
        mock_retrieve.return_value = (
            [RetrievedChunk(chunk=DocumentChunk(chunk_id="c5", document_id="doc-curr", filename="curriculum.pdf", document_type="curriculum", page_number=1, chunk_index=1, text=curriculum_evidence), score=0.95, distance=0.05)],
            [Citation(document_id="doc-curr", filename="curriculum.pdf", page_number=1, label="curriculum.pdf — Page 1", chunk_id="c5", score=0.95)],
        )

        # Turn 1: Train ML model
        r1 = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Train a classifier on feature1, feature2, target: (1,2,0), (2,3,0), (8,9,1), (9,10,1)"}],
        })
        assert r1.status_code == 200
        b1 = r1.json()
        thread_id = b1["thread_id"]
        assert b1["approval_required"] is True

        # Approve ML training
        r1_appr = client.post("/api/chat/approval", json={
            "thread_id": thread_id,
            "decision": "approved",
            "message": "Proceed with training.",
        })
        assert r1_appr.status_code == 200
        b1_appr = r1_appr.json()
        assert b1_appr["approval_required"] is False
        assert any(t.get("tool") == "train_ml_model_tool" for t in b1_appr["tool_results"])

        # Turn 2: Unrelated RAG question on the same thread
        r2 = client.post("/api/chat", json={
            "thread_id": thread_id,
            "document_id": "doc-curr",
            "messages": [{"role": "user", "content": "What does the curriculum say about Vector Databases?"}],
        })
        assert r2.status_code == 200
        b2 = r2.json()

        # Verify state isolation:
        # Turn 2 response must contain document information
        assert "Vector Databases" in b2["message"]["content"]
        assert "[curriculum.pdf — Page 1]" in b2["message"]["content"]
        # Turn 2 must NOT report ML training results, ML reports, or ML tools
        assert not any(t.get("tool") == "train_ml_model_tool" for t in b2.get("tool_results", []))
        assert not any(r.get("report_type") == "dataset_analysis" for r in b2.get("reports", []))


# ──────────────────────────────────────────────────────────────────────────────
# Scenario B: ML Train → Predict
# ──────────────────────────────────────────────────────────────────────────────

def test_scenario_b_ml_train_to_predict():
    """Turn 1: Train model → approved → trained model_id returned
    Turn 2: Predict using model → model_id reused, predict tool executes, NO retraining.
    """
    with TestClient(app) as client:
        # Turn 1: Train ML model
        r1 = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Train a classifier on feature1, feature2, churn: (1,2,0), (2,3,0), (8,9,1), (9,10,1)"}],
        })
        assert r1.status_code == 200
        b1 = r1.json()
        thread_id = b1["thread_id"]

        # Approve ML training
        r1_appr = client.post("/api/chat/approval", json={
            "thread_id": thread_id,
            "decision": "approved",
            "message": "Proceed with training.",
        })
        assert r1_appr.status_code == 200
        b1_appr = r1_appr.json()
        train_tool = next(t for t in b1_appr["tool_results"] if t.get("tool") == "train_ml_model_tool")
        trained_model_id = train_tool["result"]["model_id"]
        assert trained_model_id and trained_model_id != "N/A"

        # Turn 2: Run prediction
        r2 = client.post("/api/chat", json={
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": "Use the model you just trained to predict the churn outcome for feature1=8 and feature2=9."}],
        })
        assert r2.status_code == 200
        b2 = r2.json()

        # Must execute predict_ml_model_tool and NOT re-train
        assert b2["approval_required"] is False
        predict_tools = [t for t in b2["tool_results"] if t.get("tool") == "predict_ml_model_tool"]
        train_tools_turn2 = [t for t in b2["tool_results"] if t.get("tool") == "train_ml_model_tool"]
        assert len(predict_tools) == 1
        assert len(train_tools_turn2) == 0
        assert predict_tools[0].get("model_id") == trained_model_id


# ──────────────────────────────────────────────────────────────────────────────
# Scenario C: RAG → RAG Topic Continuation
# ──────────────────────────────────────────────────────────────────────────────

def test_scenario_c_rag_topic_continuation():
    """Turn 1: Give me 3 topics → Topics 1, 2, 3
    Turn 2: Give me more topics → Topics 4, 5, 6 (1, 2, 3 excluded)
    """
    curriculum_evidence = (
        "1. Text Foundations\n"
        "2. Embedding Basics\n"
        "3. Advanced GenAI\n"
        "4. Document Systems\n"
        "5. Vector Databases\n"
        "6. Multimodal Models\n"
    )

    with TestClient(app) as client, \
         patch("poc_kanini.rag.service.RagService.retrieve_evidence") as mock_retrieve:
        mock_retrieve.return_value = (
            [RetrievedChunk(chunk=DocumentChunk(chunk_id="c1", document_id="doc-curr", filename="curriculum.pdf", document_type="curriculum", page_number=1, chunk_index=1, text=curriculum_evidence), score=0.95, distance=0.05)],
            [Citation(document_id="doc-curr", filename="curriculum.pdf", page_number=1, label="curriculum.pdf — Page 1", chunk_id="c1", score=0.95)],
        )

        # Turn 1
        r1 = client.post("/api/chat", json={
            "document_id": "doc-curr",
            "messages": [{"role": "user", "content": "give me any 3 topics from the given curriculum"}],
        })
        assert r1.status_code == 200
        b1 = r1.json()
        thread_id = b1["thread_id"]
        c1 = b1["message"]["content"]
        assert "Text Foundations" in c1
        assert "Embedding Basics" in c1
        assert "Advanced GenAI" in c1

        # Turn 2
        r2 = client.post("/api/chat", json={
            "thread_id": thread_id,
            "document_id": "doc-curr",
            "messages": [{"role": "user", "content": "give me more topics"}],
        })
        assert r2.status_code == 200
        b2 = r2.json()
        c2 = b2["message"]["content"]
        assert "Document Systems" in c2
        assert "Vector Databases" in c2
        assert "Multimodal Models" in c2
        # Previous topics must be excluded
        assert "Text Foundations" not in c2
        assert "Embedding Basics" not in c2


# ──────────────────────────────────────────────────────────────────────────────
# Scenario D: RAG Section → Referential Follow-Up
# ──────────────────────────────────────────────────────────────────────────────

def test_scenario_d_rag_section_and_referential_followup():
    """Turn 1: What does curriculum say about Vector Databases?
    Turn 2: What are the specific technologies mentioned in that section?
    """
    curriculum_evidence = (
        "5. Vector Databases & Retrieval Systems\n"
        "5.1 Semantic Dense Representation Models\n"
        "5.2 Vector Similarity Metrics Mechanics\n"
        "5.3 FAISS Local Indexing Implementation\n"
        "5.4 ChromaDB local Cloud Setup\n"
        "5.5 Pinecone Enterprise Index Cluster\n"
        "6. Multimodal AI with Gemini\n"
    )

    with TestClient(app) as client, \
         patch("poc_kanini.rag.service.RagService.retrieve_evidence") as mock_retrieve:
        mock_retrieve.return_value = (
            [RetrievedChunk(chunk=DocumentChunk(chunk_id="c5", document_id="doc-curr", filename="curriculum.pdf", document_type="curriculum", page_number=2, chunk_index=5, text=curriculum_evidence), score=0.95, distance=0.05)],
            [Citation(document_id="doc-curr", filename="curriculum.pdf", page_number=2, label="curriculum.pdf — Page 2", chunk_id="c5", score=0.95)],
        )

        # Turn 1
        r1 = client.post("/api/chat", json={
            "document_id": "doc-curr",
            "messages": [{"role": "user", "content": "What does the curriculum say about Vector Databases & Retrieval Systems?"}],
        })
        assert r1.status_code == 200
        b1 = r1.json()
        thread_id = b1["thread_id"]
        c1 = b1["message"]["content"]
        assert "Vector Databases & Retrieval Systems" in c1
        assert "FAISS Local Indexing Implementation" in c1
        assert "Multimodal AI" not in c1  # Stopped at next section

        # Turn 2: Referential follow-up
        r2 = client.post("/api/chat", json={
            "thread_id": thread_id,
            "document_id": "doc-curr",
            "messages": [{"role": "user", "content": "What are the specific technologies or methods mentioned in that section?"}],
        })
        assert r2.status_code == 200
        b2 = r2.json()
        c2 = b2["message"]["content"]
        assert "FAISS" in c2 or "ChromaDB" in c2 or "Pinecone" in c2


# ──────────────────────────────────────────────────────────────────────────────
# Scenario E & F: Conversational Identity & Project Name Memory
# ──────────────────────────────────────────────────────────────────────────────

def test_scenario_e_and_f_identity_and_project_memory():
    """Turn 1: 'my name is Sudharsan' → Acknowledged
    Turn 2: 'what is my name?' → Sudharsan
    """
    with TestClient(app) as client:
        r1 = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "my name is Sudharsan"}],
        })
        assert r1.status_code == 200
        b1 = r1.json()
        thread_id = b1["thread_id"]
        assert "sudharsan" in b1["message"]["content"].lower()

        r2 = client.post("/api/chat", json={
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": "what is my name?"}],
        })
        assert r2.status_code == 200
        b2 = r2.json()
        assert "sudharsan" in b2["message"]["content"].lower()


# ──────────────────────────────────────────────────────────────────────────────
# Scenario G: Negative Identity Collision
# ──────────────────────────────────────────────────────────────────────────────

def test_scenario_g_negative_identity_collision():
    """'What is my project name?' must NOT be captured as 'what is my name?'."""
    from poc_kanini.graphs.supervisor import SupervisorRouter

    router = SupervisorRouter()
    q = "what is my project name?"
    decision = router._deterministic_route(q, has_documents=False)
    if decision is not None and decision.route == "general":
        assert "identity" not in decision.reason.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Scenario H: PDF Detachment
# ──────────────────────────────────────────────────────────────────────────────

def test_scenario_h_pdf_detach_in_session():
    """Attach document on Turn 1 → detach document on Turn 2 without refresh.
    Expected: Turn 2 executes without document scoping and does not retrieve doc evidence.
    """
    with TestClient(app) as client:
        # Turn 1 with document attached
        r1 = client.post("/api/chat", json={
            "document_id": "doc-curr",
            "document_ids": ["doc-curr"],
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert r1.status_code == 200
        thread_id = r1.json()["thread_id"]

        # Turn 2 with document explicitly detached (empty document_ids)
        r2 = client.post("/api/chat", json={
            "thread_id": thread_id,
            "document_id": None,
            "document_ids": [],
            "messages": [{"role": "user", "content": "hello again"}],
        })
        assert r2.status_code == 200
        b2 = r2.json()
        assert b2["citations"] == []
        assert not any(t.get("tool") == "search_document_evidence" for t in b2.get("tool_results", []))


# ──────────────────────────────────────────────────────────────────────────────
# Scenario I: Deterministic Behavior when Gemini is Unavailable
# ──────────────────────────────────────────────────────────────────────────────

def test_scenario_i_deterministic_degraded_provider():
    """When Gemini is unconfigured or unavailable, all deterministic and tool capabilities
    return clean grounded summaries with accurate provider warnings and 0 fake claims.
    """
    with patch("poc_kanini.graphs.specialists.get_settings") as mock_settings, \
         TestClient(app) as client:
        mock_settings.return_value.gemini_api_key = None
        mock_settings.return_value.environment = "test"

        r = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "what can you do?"}],
        })
        assert r.status_code == 200
        b = r.json()
        assert "document evidence" in b["message"]["content"].lower()
        assert "image analysis" in b["message"]["content"].lower()


# ──────────────────────────────────────────────────────────────────────────────
# Scenario J: Historical Report Isolation
# ──────────────────────────────────────────────────────────────────────────────

def test_scenario_j_historical_reports_isolation():
    """Turn 1: Dataset profiling request -> dataset_analysis report
    Turn 2: Casual conversation greeting -> NO reports in response.
    """
    with TestClient(app) as client:
        # Turn 1: Dataset profile
        r1 = client.post("/api/chat", json={
            "csv_data": "col1,col2\n1,2\n3,4",
            "messages": [{"role": "user", "content": "profile this dataset"}],
        })
        assert r1.status_code == 200
        b1 = r1.json()
        thread_id = b1["thread_id"]
        assert len(b1["reports"]) >= 1

        # Turn 2: Greeting on the same thread
        r2 = client.post("/api/chat", json={
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert r2.status_code == 200
        b2 = r2.json()
        # Historical report must NOT leak into Turn 2
        assert len(b2["reports"]) == 0
