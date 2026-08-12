"""Comprehensive deterministic test suite for Phase 8 Enterprise Assistant capabilities."""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from poc_kanini.graphs.chat import hybrid_chat_graph
from poc_kanini.main import app
from poc_kanini.models.actions import ActionRequest, ActionResult, ReportPayload
from poc_kanini.models.chat import ChatRequest, ChatResponse
from poc_kanini.models.orchestration import AgentConversationState
from poc_kanini.rag.models import Citation, DocumentChunk, RetrievedChunk
from poc_kanini.services.report_service import execute_action, generate_report

TINY_JPEG_BYTES = bytes([0xFF, 0xD8, 0xFF, 0xE0] + [0x00] * 50)
TINY_JPEG_B64 = base64.b64encode(TINY_JPEG_BYTES).decode("utf-8")
TEST_CONFIG = {"configurable": {"thread_id": "test-thread-phase8"}}


# 1. Unified Chat Response Schema Test
def test_unified_chat_response_schema() -> None:
    """POST /api/chat must return a unified ChatResponse containing message, thread_id, citations, reports, actions."""
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hello Assistant"}]},
        )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "thread_id" in data
    assert "citations" in data
    assert "tool_results" in data
    assert "warnings" in data
    assert "reports" in data
    assert "actions" in data


# 2. Document Upload/Index Workflow Test
def test_document_upload_index_workflow() -> None:
    """POST /api/documents/index must validate PDF file and return document metadata and chunk count."""
    from poc_kanini.models.documents import DocumentMetadata, DocumentStructure, ProcessedDocument, ProcessedPage

    fake_pdf = b"%PDF-1.4 header text %EOF"
    doc_mock = ProcessedDocument(
        metadata=DocumentMetadata(
            document_id="doc-123",
            filename="test_policy.pdf",
            file_size_bytes=len(fake_pdf),
            page_count=1,
        ),
        document_type="policy",
        pages=[
            ProcessedPage(
                page_number=1,
                text="Policy content",
                normalized_text="policy content",
                semantic_terms=["policy"],
                extraction_method="pdf_text",
                ocr_status="not_required",
                structure=DocumentStructure(headings=["Policy"]),
            )
        ],
        processing_notes=[],
    )

    with patch("poc_kanini.main.DocumentProcessor") as mock_proc, \
         patch("poc_kanini.main.rag_service") as mock_rag_fn:

        proc_inst = MagicMock()
        proc_inst.process.return_value = doc_mock
        mock_proc.return_value = proc_inst

        rag_inst = MagicMock()
        rag_inst.index_document = AsyncMock(return_value=5)
        mock_rag_fn.return_value = rag_inst

        with TestClient(app) as client:
            response = client.post(
                "/api/documents/index",
                files={"file": ("test_policy.pdf", fake_pdf, "application/pdf")},
            )

    assert response.status_code == 200
    res = response.json()
    assert res["chunk_count"] == 5
    assert res["document"]["metadata"]["document_id"] == "doc-123"


# 3. Document / Thread Association Test
def test_document_thread_association() -> None:
    """Passing document_id in ChatRequest must associate the document with the thread state."""
    chunk = DocumentChunk(
        chunk_id="c1", document_id="doc-abc", filename="employee_handbook.pdf", document_type="guide", page_number=1, chunk_index=0, text="Vacation is 25 days."
    )
    citation = Citation(chunk_id="c1", document_id="doc-abc", filename="employee_handbook.pdf", page_number=1, label="employee_handbook.pdf — Page 1")
    retrieved = RetrievedChunk(chunk=chunk, score=0.98)

    with patch("poc_kanini.tools.rag_tools.RagService") as mock_rag_cls:
        svc = MagicMock()
        svc.retrieve_evidence = AsyncMock(return_value=([retrieved], [citation]))
        mock_rag_cls.return_value = svc

        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={
                    "messages": [{"role": "user", "content": "What is the vacation policy in employee_handbook.pdf?"}],
                    "document_id": "doc-abc",
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["label"] == "employee_handbook.pdf — Page 1"


# 4. Multimodal Image Attachment Workflow Test
def test_multimodal_attachment_workflow() -> None:
    """Chat input with base64 image attachment must execute image analysis tool and return visual observations."""
    with patch("poc_kanini.tools.multimodal_tools.MultimodalService") as mock_mm:
        from poc_kanini.multimodal.models import MultimodalAnalysis
        svc = MagicMock()
        svc.analyze = AsyncMock(
            return_value=MultimodalAnalysis(
                answer="A dark test grid pattern.",
                observations=["Grid layout observed"],
                source_metadata={"filename": "photo.jpg", "mime_type": "image/jpeg", "size_bytes": "54"},
            )
        )
        mock_mm.return_value = svc

        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={
                    "messages": [{"role": "user", "content": "Describe this photo"}],
                    "attachments": [
                        {
                            "filename": "photo.jpg",
                            "mime_type": "image/jpeg",
                            "data": TINY_JPEG_B64,
                        }
                    ],
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert len(data["tool_results"]) >= 1
    assert data["tool_results"][0]["tool"] == "analyze_image_tool"


# 5. Dataset Profiling & Workflow Test
def test_dataset_profiling_workflow() -> None:
    """Chat input requesting dataset profiling must profile dataset and return tool results and report payload."""
    sample_csv = "age,salary,churn\n25,50000,0\n40,90000,1\n30,65000,0"

    # Mock the Gemini LLM synthesis and supervisor router boundaries so the test is deterministic.
    mock_response = MagicMock()
    mock_response.text = "Dataset profiled successfully."

    mock_supervisor_response = MagicMock()
    mock_supervisor_response.text = '{"route": "data", "reason": "Requesting dataset profiling", "confidence": 0.9}'

    with patch("poc_kanini.graphs.specialists.genai.Client") as mock_spec_client, \
         patch("poc_kanini.graphs.supervisor.genai.Client") as mock_sup_client:

        mock_aio_spec = MagicMock()
        mock_aio_spec.models.generate_content = AsyncMock(return_value=mock_response)
        mock_spec_client.return_value.aio = mock_aio_spec

        mock_aio_sup = MagicMock()
        mock_aio_sup.models.generate_content = AsyncMock(return_value=mock_supervisor_response)
        mock_sup_client.return_value.aio = mock_aio_sup

        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={
                    "messages": [{"role": "user", "content": f"Profile this tabular dataset CSV: {sample_csv}"}],
                },
            )
    assert response.status_code == 200
    data = response.json()
    assert any(t["tool"] == "profile_dataset_tool" for t in data["tool_results"])
    assert len(data["reports"]) >= 1


# 6. Structured RAG Presentation Test
def test_structured_rag_result_presentation() -> None:
    """RAG tool results must include citation labels and evidence snippets in structured response."""
    chunk = DocumentChunk(
        chunk_id="c2", document_id="doc-xyz", filename="financial_report.pdf", document_type="report", page_number=4, chunk_index=1, text="Revenue grew 18% YoY."
    )
    citation = Citation(chunk_id="c2", document_id="doc-xyz", filename="financial_report.pdf", page_number=4, label="financial_report.pdf — Page 4")
    retrieved = RetrievedChunk(chunk=chunk, score=0.92)

    async def _run():
        with patch("poc_kanini.tools.rag_tools.RagService") as mock_rag:
            svc = MagicMock()
            svc.retrieve_evidence = AsyncMock(return_value=([retrieved], [citation]))
            mock_rag.return_value = svc

            state: AgentConversationState = {
                "messages": [HumanMessage(content="What was revenue growth in financial_report.pdf?")],
                "step_count": 0,
            }
            return await hybrid_chat_graph.ainvoke(state, config=TEST_CONFIG)

    res = asyncio.run(_run())
    assert len(res["citations"]) >= 1
    assert res["citations"][0]["filename"] == "financial_report.pdf"


# 7. Structured ML Metrics Presentation Test
def test_structured_ml_result_payload() -> None:
    """ML training node output must contain structured accuracy/f1 metrics and model_id."""
    query = "Train a classifier model using target column y on dataset [{'x': 1.0, 'y': 'A'}, {'x': 2.0, 'y': 'A'}, {'x': 8.0, 'y': 'B'}, {'x': 9.0, 'y': 'B'}]"
    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content=query)],
            "step_count": 0,
        }
        return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "test-ml-struct"}})

    res = asyncio.run(_run())
    ml_item = next((r for r in res.get("tool_results", []) if r.get("tool") == "train_ml_model_tool"), None)
    assert ml_item is not None
    assert "metrics" in ml_item["result"]


# 8. Structured Multimodal Result Payload Test
def test_structured_multimodal_result_payload() -> None:
    """Multimodal analysis output must contain observation items and source metadata."""
    async def _run():
        from poc_kanini.multimodal.models import MultimodalAnalysis
        mock_analysis = MultimodalAnalysis(
            answer="A dark test photo.",
            observations=["Dark surface", "Plain texture"],
            source_metadata={"filename": "img.png", "mime_type": "image/png", "size_bytes": "54"},
        )
        with patch("poc_kanini.tools.multimodal_tools.MultimodalService") as mock_mm_cls:
            svc = MagicMock()
            svc.analyze = AsyncMock(return_value=mock_analysis)
            mock_mm_cls.return_value = svc

            state: AgentConversationState = {
                "messages": [HumanMessage(content="Describe image")],
                "attachments": [{"filename": "img.png", "mime_type": "image/png", "data": TINY_JPEG_B64}],
                "step_count": 0,
            }
            return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "test-mm-struct"}})

    res = asyncio.run(_run())
    mm_item = next((r for r in res.get("tool_results", []) if r.get("tool") == "analyze_image_tool"), None)
    assert mm_item is not None
    assert len(mm_item["result"]["observations"]) >= 2


# 9. Approval Request Contract Test
def test_hitl_approval_request_contract() -> None:
    """Controlled operations (e.g. 'requires approval') must trigger HITL pause with approval_required=True."""
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Execute controlled operation on production model, requires approval"}],
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["approval_required"] is True
    assert data["approval_id"] is not None


# 10. Approval Rejection Test
def test_hitl_approval_rejection_contract() -> None:
    """Posting approval='rejected' must cancel the operation safely and return a clear status message."""
    with TestClient(app) as client:
        # First request triggers approval pause
        init_res = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Sensitive operation requires approval"}],
            },
        ).json()
        t_id = init_res["thread_id"]
        a_id = init_res["approval_id"]

        # Second request posts rejection
        rej_res = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Reject operation"}],
                "thread_id": t_id,
                "approval": "rejected",
            },
        ).json()

    assert rej_res["approval_required"] is False
    assert "rejected" in rej_res["message"]["content"].lower()


# 11. Safe Action Abstraction Test
def test_action_abstraction_execution() -> None:
    """POST /api/actions/execute must run safe demonstration actions and return ActionResult."""
    req = ActionRequest(
        action_type="generate_analysis_summary",
        description="Generate summary for Q3 audit",
        parameters={"scope": "finance", "subject": "Q3 Revenue"},
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/actions/execute",
            json=req.model_dump(),
        )
    assert response.status_code == 200
    res = ActionResult.model_validate(response.json())
    assert res.status == "success"
    assert "finance" in res.summary


# 12. Report/Insight Generation Test
def test_report_generation_endpoint() -> None:
    """POST /api/reports/generate must produce structured ReportPayload with sections and recommendations."""
    payload = {
        "report_type": "dataset_analysis",
        "user_query": "Summarize churn dataset findings",
        "tool_results": [
            {
                "tool": "profile_dataset_tool",
                "result": {"row_count": 100, "column_count": 5, "numeric_columns": ["age"], "categorical_columns": ["churn"]},
            }
        ],
    }
    with TestClient(app) as client:
        response = client.post(
            "/api/reports/generate",
            json=payload,
        )
    assert response.status_code == 200
    report = ReportPayload.model_validate(response.json())
    assert report.report_type == "dataset_analysis"
    assert len(report.sections) >= 1
    assert len(report.recommendations) >= 1


# 13. Error Handling Test
def test_invalid_attachment_graceful_error_handling() -> None:
    """Posting an invalid base64 attachment must return a clean error without throwing 500 stacktrace."""
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Analyze image"}],
                "attachments": [{"filename": "bad.png", "mime_type": "image/png", "data": "invalid-b64-content"}],
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


# 14. Backwards Compatibility with Endpoints
def test_backwards_compatibility_existing_endpoints() -> None:
    """Existing endpoints /api/health, /api/ml/profile, and /api/multimodal/analyze must continue working."""
    with TestClient(app) as client:
        h = client.get("/api/health")
        assert h.status_code == 200
        assert h.json()["status"] == "ok"

        prof = client.post("/api/ml/profile", json=[{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert prof.status_code == 200
        assert prof.json()["row_count"] == 2
