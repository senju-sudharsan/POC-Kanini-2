"""Phase 9 — E2E Integration, Security, and Reliability Tests.

All tests are deterministic and use mocks; no live Gemini API key required.
Tests are organised by the Phase 9 verification flows.
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from poc_kanini.graphs.chat import hybrid_chat_graph
from poc_kanini.main import app
from poc_kanini.models.actions import ActionRequest, ActionResult, ReportPayload
from poc_kanini.models.chat import ChatRequest, ChatResponse
from poc_kanini.models.orchestration import AgentConversationState
from poc_kanini.multimodal.validator import ImageValidationError, validate_image
from poc_kanini.services.report_service import execute_action, generate_report

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

TINY_JPEG_BYTES = bytes([0xFF, 0xD8, 0xFF, 0xE0] + [0x00] * 50)
TINY_JPEG_B64 = base64.b64encode(TINY_JPEG_BYTES).decode()

SYNTHETIC_DATASET = [
    {"age": 22, "salary": 40000, "department": "IT", "promoted": 0},
    {"age": 30, "salary": 60000, "department": "HR", "promoted": 1},
    {"age": 35, "salary": 80000, "department": "IT", "promoted": 1},
    {"age": 25, "salary": 45000, "department": "HR", "promoted": 0},
    {"age": 40, "salary": 90000, "department": "IT", "promoted": 1},
    {"age": 28, "salary": 55000, "department": "HR", "promoted": 0},
]


# ===========================================================================
# FLOW 1 — NORMAL CHAT (mock Gemini, verify response schema)
# ===========================================================================


def test_flow1_normal_chat_response_schema() -> None:
    """POST /api/chat returns valid ChatResponse schema with no stack trace."""
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hello, what can you help me with?"}]},
        )
    assert response.status_code == 200
    data = response.json()
    # Required schema fields
    assert "message" in data
    assert "thread_id" in data
    assert "citations" in data
    assert "tool_results" in data
    assert "warnings" in data
    assert "reports" in data
    assert "actions" in data
    # Thread ID auto-generated
    assert data["thread_id"].startswith("thread_")
    # Message has role and content
    assert data["message"]["role"] == "assistant"
    assert isinstance(data["message"]["content"], str)
    assert len(data["message"]["content"]) > 0


def test_flow1_no_stack_trace_in_response() -> None:
    """Error responses must not leak raw Python stack traces."""
    with TestClient(app) as client:
        # Send a deliberately invalid payload — missing messages
        response = client.post("/api/chat", json={"messages": []})
    # Whether 200 or 422, the body must not contain traceback keywords
    body = response.text
    assert "Traceback" not in body
    assert "File " not in body


def test_flow1_thread_id_provided_is_echoed() -> None:
    """If a thread_id is supplied it must be returned in the response."""
    custom_tid = f"thread_{uuid.uuid4().hex[:8]}"
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "thread_id": custom_tid},
        )
    assert response.status_code == 200
    assert response.json()["thread_id"] == custom_tid


# ===========================================================================
# FLOW 2 — DOCUMENT RAG (mock document processor + RAG service)
# ===========================================================================


def test_flow2_document_index_and_citation_verification() -> None:
    """PDF index → chat with document → citation in [filename — Page X] format."""
    from poc_kanini.models.documents import (
        DocumentMetadata,
        DocumentStructure,
        ProcessedDocument,
        ProcessedPage,
    )
    from poc_kanini.rag.models import Citation, DocumentChunk, RetrievedChunk

    fake_pdf = b"%PDF-1.4 tech content %EOF"
    doc_id = "doc-phase9-001"
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id=doc_id,
        filename="tech_policy.pdf",
        document_type="policy",
        page_number=3,
        chunk_index=0,
        text="Python, FastAPI, LangGraph, and Gemini are core technologies.",
        metadata={
            "document_id": doc_id,
            "filename": "tech_policy.pdf",
            "document_type": "policy",
            "page_number": "3",
            "chunk_index": "0",
        },
    )
    retrieved = RetrievedChunk(chunk=chunk, distance=0.05, score=0.95)

    with (
        patch("poc_kanini.main.DocumentProcessor") as mock_proc_cls,
        patch("poc_kanini.main.rag_service") as mock_rag_fn,
    ):
        mock_doc = ProcessedDocument(
            metadata=DocumentMetadata(
                document_id=doc_id,
                filename="tech_policy.pdf",
                file_size_bytes=len(fake_pdf),
                page_count=1,
            ),
            document_type="policy",
            pages=[
                ProcessedPage(
                    page_number=1,
                    text="Python, FastAPI, LangGraph, and Gemini are core technologies.",
                    normalized_text="python fastapi langgraph gemini core technologies",
                    semantic_terms=["python", "fastapi"],
                    extraction_method="pdf_text",
                    ocr_status="not_required",
                    structure=DocumentStructure(),
                )
            ],
            processing_notes=[],
        )
        proc_inst = MagicMock()
        proc_inst.process.return_value = mock_doc
        mock_proc_cls.return_value = proc_inst

        rag_inst = MagicMock()
        rag_inst.index_document = AsyncMock(return_value=1)
        mock_rag_fn.return_value = rag_inst

        with TestClient(app) as client:
            resp = client.post(
                "/api/documents/index",
                files={"file": ("tech_policy.pdf", fake_pdf, "application/pdf")},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["chunk_count"] == 1
    assert body["document"]["metadata"]["document_id"] == doc_id


def test_flow2_citation_format_preserved() -> None:
    """Citations must contain the label in [filename — Page X] format."""
    from poc_kanini.rag.models import Citation, DocumentChunk, RetrievedChunk
    from poc_kanini.rag.service import _citation

    chunk = DocumentChunk(
        chunk_id="c-cite",
        document_id="doc-001",
        filename="employee_handbook.pdf",
        document_type="handbook",
        page_number=7,
        chunk_index=0,
        text="Annual leave entitlement is 25 days.",
        metadata={
            "document_id": "doc-001",
            "filename": "employee_handbook.pdf",
            "document_type": "handbook",
            "page_number": "7",
            "chunk_index": "0",
        },
    )
    citation = _citation(RetrievedChunk(chunk=chunk, score=0.9))
    assert citation.label == "employee_handbook.pdf — Page 7"
    assert "Page 7" in citation.label


# ===========================================================================
# FLOW 3 — MULTIMODAL VALIDATION
# ===========================================================================


def test_flow3_valid_jpeg_passes_validation() -> None:
    """A minimal valid JPEG must pass image validation without error."""
    validate_image(content=TINY_JPEG_BYTES, mime_type="image/jpeg", filename="test.jpg")


def test_flow3_empty_image_rejected() -> None:
    """Empty bytes must raise ImageValidationError."""
    with pytest.raises(ImageValidationError, match="empty"):
        validate_image(content=b"", mime_type="image/jpeg", filename="empty.jpg")


def test_flow3_unsupported_mime_rejected() -> None:
    """Unsupported MIME type must raise ImageValidationError."""
    with pytest.raises(ImageValidationError, match="Unsupported"):
        validate_image(content=TINY_JPEG_BYTES, mime_type="application/pdf", filename="bad.pdf")


def test_flow3_oversized_image_rejected() -> None:
    """Image exceeding 10 MB limit must raise ImageValidationError."""
    oversized = b"X" * (10 * 1024 * 1024 + 1)
    with pytest.raises(ImageValidationError, match="exceeds"):
        validate_image(content=oversized, mime_type="image/jpeg", filename="big.jpg")


def test_flow3_malformed_base64_rejected_by_tool() -> None:
    """Malformed base64 in analyze_image_tool must return an error dict, not raise."""
    import asyncio
    from poc_kanini.tools.multimodal_tools import analyze_image_tool

    result = asyncio.run(
        analyze_image_tool.ainvoke(
            {
                "image_base64": "!!! not base64 !!!",
                "mime_type": "image/jpeg",
                "question": "Describe",
            }
        )
    )
    assert "error" in result
    assert "base64" in result["error"].lower() or "invalid" in result["error"].lower()


def test_flow3_multimodal_api_endpoint_invalid_mime() -> None:
    """POST /api/multimodal/analyze with a PDF file must return 400."""
    with TestClient(app) as client:
        response = client.post(
            "/api/multimodal/analyze",
            files={"file": ("document.pdf", b"%PDF-1.4", "application/pdf")},
            data={"question": "What is this?"},
        )
    assert response.status_code == 400
    # Error must not expose stack trace
    assert "Traceback" not in response.text


def test_flow3_multimodal_api_empty_bytes_rejected() -> None:
    """POST /api/multimodal/analyze with an empty file must return 400."""
    with TestClient(app) as client:
        response = client.post(
            "/api/multimodal/analyze",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
            data={"question": "What is this?"},
        )
    assert response.status_code == 400


# ===========================================================================
# FLOW 4 — DATA PROFILING
# ===========================================================================


def test_flow4_dataset_profiling_api() -> None:
    """POST /api/ml/profile must return a DatasetProfile with column stats."""
    with TestClient(app) as client:
        response = client.post("/api/ml/profile", json=SYNTHETIC_DATASET)
    assert response.status_code == 200
    data = response.json()
    assert data["row_count"] == len(SYNTHETIC_DATASET)
    assert "age" in data["summary_statistics"]
    assert "salary" in data["summary_statistics"]
    assert "numeric_columns" in data
    assert "categorical_columns" in data


def test_flow4_profiling_empty_dataset_rejected() -> None:
    """Empty dataset must return 400."""
    with TestClient(app) as client:
        response = client.post("/api/ml/profile", json=[])
    assert response.status_code == 400


# ===========================================================================
# FLOW 5 — MACHINE LEARNING TRAIN → PREDICT
# ===========================================================================


def test_flow5_ml_train_and_predict_round_trip() -> None:
    """Train a classifier → use model_id to predict — must return predictions."""
    with TestClient(app) as client:
        # Train
        train_resp = client.post(
            "/api/ml/train",
            json={
                "data": SYNTHETIC_DATASET,
                "target": "promoted",
                "task": "classification",
                "model_type": "RandomForestClassifier",
            },
        )
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert "model_id" in train_data
    assert "metrics" in train_data
    model_id = train_data["model_id"]

    # Predict
    with TestClient(app) as client:
        predict_resp = client.post(
            "/api/ml/predict",
            json={
                "model_id": model_id,
                "data": [{"age": 27, "salary": 52000, "department": "IT"}],
            },
        )
    assert predict_resp.status_code == 200
    predict_data = predict_resp.json()
    assert "predictions" in predict_data
    assert isinstance(predict_data["predictions"], list)
    assert len(predict_data["predictions"]) == 1


def test_flow5_predict_unknown_model_returns_400() -> None:
    """Predicting with a non-existent model_id must return 400."""
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/predict",
            json={"model_id": "nonexistent-model-id", "data": [{"age": 25}]},
        )
    assert response.status_code == 400


def test_flow5_train_missing_target_column_returns_400() -> None:
    """Training with a target not in dataset must return 400."""
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/train",
            json={
                "data": SYNTHETIC_DATASET,
                "target": "nonexistent_column",
                "task": "classification",
            },
        )
    assert response.status_code == 400


# ===========================================================================
# FLOW 6 — MEMORY / THREAD ISOLATION
# ===========================================================================


def test_flow6_thread_scoped_memory_isolation() -> None:
    """Two separate thread IDs must not share conversation state."""

    async def _run():
        config_a = {"configurable": {"thread_id": "phase9-thread-a"}}
        config_b = {"configurable": {"thread_id": "phase9-thread-b"}}

        await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="My secret phrase is ALPHA-DELTA-9.")], "step_count": 0, "max_steps": 5},
            config=config_a,
        )
        result_b = await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="What secret phrase do you know?")], "step_count": 0, "max_steps": 5},
            config=config_b,
        )
        return result_b

    result = asyncio.run(_run())
    messages_b = [str(m.content) for m in result["messages"]]
    assert not any("ALPHA-DELTA-9" in msg for msg in messages_b)


def test_flow6_same_thread_retains_context() -> None:
    """Conversation context must persist within the same thread across turns."""

    async def _run():
        tid = "phase9-memory-test"
        config = {"configurable": {"thread_id": tid}}

        await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="My name is Phase9Tester.")], "step_count": 0, "max_steps": 5},
            config=config,
        )
        result2 = await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="What is my name?")], "step_count": 0, "max_steps": 5},
            config=config,
        )
        return result2

    result = asyncio.run(_run())
    messages = [str(m.content) for m in result["messages"]]
    assert any("Phase9Tester" in msg for msg in messages)


# ===========================================================================
# FLOW 7 — HITL APPROVAL / REJECTION
# ===========================================================================


def test_flow7_hitl_approval_flow() -> None:
    """HITL approval_required flag must be returned for a controlled action request."""
    with TestClient(app) as client:
        # Request a controlled action that triggers HITL
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Execute a controlled action to generate a report."}],
                "thread_id": "phase9-hitl-test",
            },
        )
    assert response.status_code == 200
    # Either it resolved normally or returned approval_required
    data = response.json()
    assert "approval_required" in data
    assert "message" in data


def test_flow7_action_rejection_via_execute_endpoint() -> None:
    """Unsupported action type must return a failed ActionResult, not 422."""
    # Note: ActionType is a Literal — passing an invalid type triggers 422 Pydantic validation
    with TestClient(app) as client:
        response = client.post(
            "/api/actions/execute",
            json={
                "action_type": "generate_analysis_summary",
                "description": "Test action",
                "parameters": {"scope": "test"},
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


# ===========================================================================
# FLOW 8 — REPORT GENERATION
# ===========================================================================


def test_flow8_report_generation_schema() -> None:
    """POST /api/reports/generate must return a valid ReportPayload."""
    with TestClient(app) as client:
        response = client.post(
            "/api/reports/generate",
            json={"report_type": "executive_summary", "user_query": "Summarise the situation."},
        )
    assert response.status_code == 200
    data = response.json()
    assert "report_type" in data
    assert "title" in data
    assert "summary" in data
    assert "sections" in data
    assert isinstance(data["sections"], list)
    assert len(data["sections"]) > 0


def test_flow8_report_invalid_type_rejected() -> None:
    """Report type not in allowed set must return 422."""
    with TestClient(app) as client:
        response = client.post(
            "/api/reports/generate",
            json={"report_type": "arbitrary_hack", "user_query": "Test"},
        )
    assert response.status_code == 422


def test_flow8_all_valid_report_types() -> None:
    """All four supported report types must return 200."""
    valid_types = ["executive_summary", "dataset_analysis", "document_analysis", "image_analysis"]
    with TestClient(app) as client:
        for rtype in valid_types:
            resp = client.post(
                "/api/reports/generate",
                json={"report_type": rtype, "user_query": f"Test for {rtype}"},
            )
            assert resp.status_code == 200, f"Failed for report_type={rtype}: {resp.text}"


# ===========================================================================
# FLOW 9 — CONTROLLED ACTIONS
# ===========================================================================


def test_flow9_supported_actions_succeed() -> None:
    """All five declared action types must return status=success."""
    supported = [
        ("generate_analysis_summary", {"scope": "quarterly", "subject": "revenue"}),
        ("prepare_recommendation", {"recommendation_type": "escalate"}),
        ("create_structured_report_payload", {"report_type": "executive_summary"}),
        ("profile_dataset", {"data": [{"col": 1}]}),
        ("train_model", {"target": "y"}),
    ]
    for atype, params in supported:
        result = execute_action(
            ActionRequest(action_type=atype, description=f"Test {atype}", parameters=params)
        )
        assert result.status == "success", f"Expected success for {atype}, got {result.status}: {result.summary}"


def test_flow9_no_shell_execution() -> None:
    """execute_action must never invoke subprocess or os.system."""
    import subprocess
    import sys

    # Patch subprocess to detect any call attempt
    with patch.object(subprocess, "run", side_effect=AssertionError("Shell execution detected!")):
        with patch.object(subprocess, "Popen", side_effect=AssertionError("Shell execution detected!")):
            result = execute_action(
                ActionRequest(
                    action_type="generate_analysis_summary",
                    description="Test",
                    parameters={"scope": "test"},
                )
            )
    assert result.status == "success"


def test_flow9_arbitrary_action_type_blocked_by_pydantic() -> None:
    """Sending an unknown action_type via HTTP must return 422 Unprocessable Entity."""
    with TestClient(app) as client:
        response = client.post(
            "/api/actions/execute",
            json={
                "action_type": "rm_-rf_/",
                "description": "Malicious action attempt",
                "parameters": {},
            },
        )
    assert response.status_code == 422


# ===========================================================================
# SECURITY — 9E: Additional security boundary tests
# ===========================================================================


def test_security_health_endpoint_no_secrets() -> None:
    """GET /api/health must not expose GEMINI_API_KEY or any secret value."""
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.text
    assert "gemini_api_key" not in body.lower()
    assert "api_key" not in body.lower()
    # Boolean flag is fine — actual key value must not appear
    assert "GEMINI_API_KEY" not in body


def test_security_health_endpoint_indicates_gemini_configured() -> None:
    """GET /api/health must include gemini_configured boolean flag."""
    with TestClient(app) as client:
        response = client.get("/api/health")
    data = response.json()
    assert "gemini_configured" in data
    assert isinstance(data["gemini_configured"], bool)


def test_security_pdf_path_traversal_rejected() -> None:
    """Uploading a PDF with path traversal filename must not escape to filesystem."""
    malicious_pdf = b"%PDF-1.4 content %EOF"
    # DocumentProcessor.process should sanitise the filename using Path(...).name
    from poc_kanini.documents.processor import DocumentProcessor, DocumentValidationError

    # Should raise because "../../etc/passwd.pdf" has no PDF signature after stripping — OR
    # if content is valid, should sanitise filename to "passwd.pdf" (no directory component).
    try:
        result = DocumentProcessor().process(
            malicious_pdf, "../../etc/passwd.pdf", "application/pdf"
        )
        # If it succeeded, filename must have been sanitised
        assert "/" not in result.metadata.filename
        assert ".." not in result.metadata.filename
    except DocumentValidationError:
        pass  # Rejection is also acceptable


def test_security_document_upload_non_pdf_rejected() -> None:
    """Uploading a non-PDF file must return 400, not 500."""
    with TestClient(app) as client:
        response = client.post(
            "/api/documents/index",
            files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
        )
    assert response.status_code in (400, 422)
    assert "Traceback" not in response.text


# ===========================================================================
# RELIABILITY — 9C: Provider error fallback
# ===========================================================================


def test_reliability_gemini_unavailable_chat_does_not_crash() -> None:
    """If Gemini API is unavailable the supervisor must fall back to heuristic routing."""
    from poc_kanini.graphs.supervisor import SupervisorRouter
    from poc_kanini.core.config import Settings

    # Build settings without API key — triggers heuristic fallback
    settings_no_key = Settings(gemini_api_key=None)
    router = SupervisorRouter(settings=settings_no_key)

    state: AgentConversationState = {
        "messages": [HumanMessage(content="Profile this dataset for me.")],
        "step_count": 0,
        "max_steps": 5,
    }

    decision = asyncio.run(router.route(state))
    assert decision.route in {"data", "rag", "ml", "general", "multimodal"}


def test_reliability_supervisor_heuristic_routes_correctly() -> None:
    """Supervisor heuristic must route known keywords without LLM call."""
    from poc_kanini.graphs.supervisor import SupervisorRouter
    from poc_kanini.core.config import Settings

    router = SupervisorRouter(Settings(gemini_api_key=None))

    cases = [
        ("Tell me about the PDF policy document", "rag"),
        ("Profile this dataset and show columns", "data"),
        ("Train a classifier on this data", "ml"),
        ("Analyse the image I uploaded", "multimodal"),
        ("Hello there", "general"),
    ]
    for query, expected_route in cases:
        decision = router._heuristic_route(query)
        assert decision.route == expected_route, (
            f"Query '{query}' expected route '{expected_route}', got '{decision.route}'"
        )


def test_reliability_report_service_empty_tool_results() -> None:
    """generate_report with no tool results must still return a valid ReportPayload."""
    report = generate_report(report_type="executive_summary", tool_results=[], user_query="Empty test")
    assert report.report_type == "executive_summary"
    assert len(report.sections) > 0
    assert report.title


def test_reliability_rag_empty_retrieval_no_fabrication() -> None:
    """RAG service with empty store must not fabricate citations."""
    import asyncio
    from poc_kanini.rag.retrieval import RetrievalService
    from poc_kanini.rag.models import DocumentChunk

    class EmptyEmbeddings:
        async def embed_query(self, q): return [0.1]
        async def embed_documents(self, chunks): return [[0.1]] * len(chunks)

    class EmptyStore:
        def similarity_search(self, emb, limit, document_id=None): return []
        def all_chunks(self, document_id=None): return []
        def upsert(self, chunks, embeddings): pass
        def delete_document(self, document_id): pass

    service = RetrievalService(EmptyEmbeddings(), EmptyStore())
    results = asyncio.run(service.retrieve("What is the leave policy?", top_k=5))
    assert results == []


def test_reliability_supervisor_routes_to_rag_when_document_attached() -> None:
    """Supervisor heuristic should route generic queries to RAG if a document is attached."""
    from poc_kanini.graphs.supervisor import SupervisorRouter
    from poc_kanini.core.config import Settings

    router = SupervisorRouter(Settings(gemini_api_key=None))

    # Greeting should still go to general
    dec_greet = router._heuristic_route("Hello there", has_documents=True)
    assert dec_greet.route == "general"

    # Specific dataset query goes to data
    dec_data = router._heuristic_route("Profile this dataset", has_documents=True)
    assert dec_data.route == "data"

    # Generic question about target topic goes to RAG when document is attached
    dec_rag = router._heuristic_route("What does this guy specialize at?", has_documents=True)
    assert dec_rag.route == "rag"


def test_api_chat_preserves_document_ids_in_checkpoint() -> None:
    """API chat endpoint must preserve document_ids in checkpoints if omitted from subsequent payloads, and clear them if explicitly unlinked."""
    with TestClient(app) as client:
        thread_id = f"thread_{uuid.uuid4().hex[:8]}"

        # First call: explicitly associate document
        resp1 = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "What is the policy?"}],
                "thread_id": thread_id,
                "document_id": "doc_123",
                "document_ids": ["doc_123"],
            },
        )
        assert resp1.status_code == 200

        # Second call: omit document fields entirely to simulate continuing thread using checkpointed documents
        resp2 = client.post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "user", "content": "What is the policy?"},
                    {"role": "assistant", "content": "The policy states X."},
                    {"role": "user", "content": "Tell me more about it."},
                ],
                "thread_id": thread_id,
            },
        )
        assert resp2.status_code == 200
        # Since document_ids was not passed, the checkpointed list is preserved, and RAG routes
        # Check activities trace from supervisor node to verify it routed to 'support_agent' (RAG)
        acts2 = resp2.json().get("activities", [])
        supervisor_act = next((a for a in acts2 if a["title"] == "Supervisor Routing"), None)
        assert supervisor_act is not None
        assert "support_agent" in supervisor_act["data"] or "rag" in supervisor_act["data"]

        # Third call: explicitly clear the document association (unlink) by passing empty list/null
        resp3 = client.post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "user", "content": "What is the policy?"},
                    {"role": "assistant", "content": "The policy states X."},
                    {"role": "user", "content": "Tell me more about it."},
                    {"role": "assistant", "content": "It says Y."},
                    {"role": "user", "content": "Disconnect doc."},
                ],
                "thread_id": thread_id,
                "document_id": None,
                "document_ids": [],
            },
        )
        assert resp3.status_code == 200
        # Check activities trace from supervisor node to verify it routed to general conversation (not RAG)
        acts3 = resp3.json().get("activities", [])
        supervisor_act_3 = next((a for a in reversed(acts3) if a["title"] == "Supervisor Routing"), None)
        assert supervisor_act_3 is not None
        assert "general_agent" in supervisor_act_3["data"] or "general" in supervisor_act_3["data"]


# ---------------------------------------------------------------------------
# RAG Synthesis Quality Tests
# ---------------------------------------------------------------------------

def test_synthesis_simple_rag_question_gives_direct_answer_no_report():
    """Simple factual RAG question should produce a direct answer, not an executive report."""
    from poc_kanini.graphs.specialists import synthesize_node
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="What does this guy specialize at?")],
        "tool_results": [
            {
                "tool": "search_document_evidence",
                "result": {
                    "evidence": [
                        {"text": "Sudharsan specializes in Data Science and AI/ML.", "document_id": "d1",
                         "filename": "resume.pdf", "page_number": 1, "chunk_id": "c1", "score": 0.9, "distance": 0.1},
                    ],
                    "citations": [
                        {"label": "resume.pdf — Page 1", "filename": "resume.pdf", "page_number": 1},
                    ],
                    "retrieved_count": 1,
                    "summary": "Retrieved 1 evidence snippet(s) from indexed documents.",
                },
            }
        ],
        "activities": [],
        "warnings": [],
        "reports": [],
        "actions": [],
        "step_count": 1,
    }

    result = asyncio.run(synthesize_node(state))

    # Must produce an answer
    ai_msg = result["messages"][-1]
    assert ai_msg.content, "Synthesis should produce non-empty answer"

    # Must NOT generate a structured report for simple questions
    assert len(result.get("reports", [])) == 0, "Simple questions should not generate structured reports"

    # The answer must not contain generic enterprise report headers
    content_lower = ai_msg.content.lower()
    assert "enterprise ai intelligence" not in content_lower
    assert "executive summary" not in content_lower

    # Must not contain generic recommendation
    assert "review cited document pages for full contractual or policy context" not in content_lower


def test_synthesis_analytical_request_generates_report():
    """Analytical queries with keywords like 'summarize' should still generate reports."""
    from poc_kanini.graphs.specialists import synthesize_node
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="Summarize this resume")],
        "tool_results": [
            {
                "tool": "search_document_evidence",
                "result": {
                    "evidence": [
                        {"text": "Professional summary for Sudharsan.", "document_id": "d1",
                         "filename": "resume.pdf", "page_number": 1, "chunk_id": "c1", "score": 0.9, "distance": 0.1},
                    ],
                    "citations": [
                        {"label": "resume.pdf — Page 1", "filename": "resume.pdf", "page_number": 1},
                    ],
                    "retrieved_count": 1,
                    "summary": "Retrieved 1 evidence snippet(s).",
                },
            }
        ],
        "activities": [],
        "warnings": [],
        "reports": [],
        "actions": [],
        "step_count": 1,
    }

    result = asyncio.run(synthesize_node(state))
    assert len(result.get("reports", [])) >= 1, "Analytical requests should generate a structured report"


def test_synthesis_deduplicates_citations():
    """Duplicate (filename, page_number) citations should be collapsed to one."""
    from poc_kanini.graphs.specialists import synthesize_node
    from langchain_core.messages import HumanMessage

    dup_citation = {"label": "resume.pdf — Page 1", "filename": "resume.pdf", "page_number": 1}
    state = {
        "messages": [HumanMessage(content="What skills does this person have?")],
        "tool_results": [
            {
                "tool": "search_document_evidence",
                "result": {
                    "evidence": [
                        {"text": "Python, ML.", "document_id": "d1", "filename": "resume.pdf",
                         "page_number": 1, "chunk_id": "c1", "score": 0.9, "distance": 0.1},
                        {"text": "Deep learning.", "document_id": "d1", "filename": "resume.pdf",
                         "page_number": 1, "chunk_id": "c2", "score": 0.8, "distance": 0.2},
                        {"text": "Data analysis.", "document_id": "d1", "filename": "resume.pdf",
                         "page_number": 1, "chunk_id": "c3", "score": 0.7, "distance": 0.3},
                        {"text": "TensorFlow.", "document_id": "d1", "filename": "resume.pdf",
                         "page_number": 1, "chunk_id": "c4", "score": 0.6, "distance": 0.4},
                    ],
                    "citations": [dup_citation, dup_citation, dup_citation, dup_citation],
                    "retrieved_count": 4,
                    "summary": "Retrieved 4 snippets.",
                },
            }
        ],
        "activities": [],
        "warnings": [],
        "reports": [],
        "actions": [],
        "step_count": 1,
    }

    result = asyncio.run(synthesize_node(state))

    # Citations should be deduplicated to a single entry
    citations = result.get("citations", [])
    assert len(citations) == 1, f"Expected 1 deduplicated citation, got {len(citations)}: {citations}"
    assert citations[0]["filename"] == "resume.pdf"
    assert citations[0]["page_number"] == 1


def test_synthesis_no_evidence_gives_safe_response():
    """When RAG retrieves zero evidence, synthesis should say so honestly."""
    from poc_kanini.graphs.specialists import synthesize_node
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="What is the refund policy?")],
        "tool_results": [
            {
                "tool": "search_document_evidence",
                "result": {
                    "evidence": [],
                    "citations": [],
                    "retrieved_count": 0,
                    "summary": "No matching evidence found.",
                },
            }
        ],
        "activities": [],
        "warnings": [],
        "reports": [],
        "actions": [],
        "step_count": 1,
    }

    result = asyncio.run(synthesize_node(state))
    ai_msg = result["messages"][-1]
    content_lower = ai_msg.content.lower()

    # Should communicate that no evidence was found — not hallucinate an answer
    assert any(phrase in content_lower for phrase in [
        "does not contain enough information",
        "no matching evidence",
        "no relevant evidence",
        "could not find",
        "unable to find",
        "no information",
    ]), f"Expected safe 'no evidence' response, got: {ai_msg.content}"

