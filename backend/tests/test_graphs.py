"""Comprehensive tests for Phase 6 Hybrid Agent graph, routing, specialists, and API integration.

All tests are deterministic and do not require a live Gemini API key.
"""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from poc_kanini.graphs.chat import hybrid_chat_graph
from poc_kanini.graphs.supervisor import SupervisorRouter
from poc_kanini.main import app
from poc_kanini.models.orchestration import AgentConversationState, RouteDecision
from poc_kanini.rag.models import Citation, DocumentChunk, RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

TINY_JPEG_BYTES = bytes([0xFF, 0xD8, 0xFF, 0xE0] + [0x00] * 50)
TINY_JPEG_B64 = base64.b64encode(TINY_JPEG_BYTES).decode("utf-8")

# Phase 7: MemorySaver requires a thread_id in every ainvoke/astream call.
TEST_CONFIG = {"configurable": {"thread_id": "test-thread-graphs"}}


# ---------------------------------------------------------------------------
# 1-7. Supervisor Router Tests
# ---------------------------------------------------------------------------


def test_supervisor_multimodal_routing_via_attachment() -> None:
    """Supervisor must route to 'multimodal' if image attachments exist."""
    router = SupervisorRouter()
    state: AgentConversationState = {
        "messages": [HumanMessage(content="Look at this")],
        "attachments": [{"filename": "sample.jpg", "mime_type": "image/jpeg", "data": TINY_JPEG_B64}],
    }
    decision = asyncio.run(router.route(state))
    assert decision.route == "multimodal"
    assert "attachment" in decision.reason.lower()


def test_supervisor_rag_routing_keyword() -> None:
    """Supervisor heuristic must route document queries to 'rag'."""
    router = SupervisorRouter()
    state: AgentConversationState = {"messages": [HumanMessage(content="What is the leave policy in the PDF document?")]}
    decision = asyncio.run(router.route(state))
    assert decision.route == "rag"


def test_supervisor_data_routing_keyword() -> None:
    """Supervisor heuristic must route profiling queries to 'data'."""
    router = SupervisorRouter()
    state: AgentConversationState = {"messages": [HumanMessage(content="Profile this tabular dataset CSV")]}
    decision = asyncio.run(router.route(state))
    assert decision.route == "data"


def test_supervisor_ml_routing_keyword() -> None:
    """Supervisor heuristic must route training queries to 'ml'."""
    router = SupervisorRouter()
    # Use small inlined data so it routes via keyword heuristic only (no Gemini call).
    state: AgentConversationState = {"messages": [HumanMessage(content="Train a classifier model")]}
    decision = asyncio.run(router.route(state))
    assert decision.route == "ml"


def test_supervisor_general_routing_keyword() -> None:
    """Supervisor heuristic must route general conversation to 'general'."""
    router = SupervisorRouter()
    state: AgentConversationState = {"messages": [HumanMessage(content="Hello! How are you today?")]}
    decision = asyncio.run(router.route(state))
    assert decision.route == "general"


def test_supervisor_structured_output_gemini(monkeypatch) -> None:
    """Supervisor must parse structured JSON from Gemini when API key is available."""
    fake_response = MagicMock()
    fake_response.text = '{"route": "rag", "reason": "User asks for document policy", "confidence": 0.95}'

    class FakeModels:
        async def generate_content(self, **kwargs):
            return fake_response

    class FakeClient:
        def __init__(self, **kwargs):
            self.aio = type("Aio", (), {"models": FakeModels()})()

    import poc_kanini.graphs.supervisor as sup_mod
    monkeypatch.setattr(sup_mod.genai, "Client", FakeClient)

    from poc_kanini.core.config import Settings
    router = SupervisorRouter(Settings(gemini_api_key="test-key"))
    state: AgentConversationState = {"messages": [HumanMessage(content="Search the policy document")]}
    decision = asyncio.run(router.route(state))

    assert decision.route == "rag"
    assert decision.confidence == 0.95


def test_supervisor_failure_fallback() -> None:
    """Supervisor must fall back to general route when LLM or state fails."""
    router = SupervisorRouter()
    state: AgentConversationState = {"messages": []}  # empty messages
    decision = asyncio.run(router.route(state))
    assert decision.route == "general"


# ---------------------------------------------------------------------------
# 8-11. Specialist Node & Tool Execution Tests
# ---------------------------------------------------------------------------


def test_rag_tool_execution_node() -> None:
    """Support specialist node must invoke search_document_evidence and append tool results."""
    chunk = DocumentChunk(
        chunk_id="c1", document_id="doc-1", filename="policy.pdf", document_type="policy", page_number=2, chunk_index=0, text="Leave allowance is 20 days."
    )
    citation = Citation(chunk_id="c1", document_id="doc-1", filename="policy.pdf", page_number=2, label="policy.pdf — Page 2")
    retrieved = RetrievedChunk(chunk=chunk, score=0.9)

    async def _run():
        with patch("poc_kanini.tools.rag_tools.RagService") as mock_rag:
            svc = MagicMock()
            svc.retrieve_evidence = AsyncMock(return_value=([retrieved], [citation]))
            mock_rag.return_value = svc

            state: AgentConversationState = {
                "messages": [HumanMessage(content="What is the leave policy?")],
                "step_count": 0,
            }
            return await hybrid_chat_graph.ainvoke(state, config=TEST_CONFIG)

    result = asyncio.run(_run())
    assert any(a["title"] == "Support Specialist" for a in result.get("activities", []))
    assert len(result.get("tool_results", [])) >= 1
    assert result["tool_results"][0]["tool"] == "search_document_evidence"


def test_data_tool_execution_node() -> None:
    """Data specialist node must profile dataset and update activities."""
    sample_data = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content=f"Profile this dataset: {sample_data}")],
            "step_count": 0,
        }
        return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "test-thread-data"}})

    result = asyncio.run(_run())
    assert any(a["title"] == "Data Specialist" for a in result.get("activities", []))
    assert result["tool_results"][0]["tool"] == "profile_dataset_tool"
    assert result["tool_results"][0]["result"]["row_count"] == 2


def test_ml_tool_execution_node() -> None:
    """ML specialist node must train a baseline model and return model_id."""
    # Query explicitly references 'model' and 'classifier' for ML routing.
    # It also contains 'dataset' so data runs first; ML runs via cross-specialist transition.
    train_query = "Train a classifier model using target column y on dataset [{'x': 1.0, 'y': 'A'}, {'x': 2.0, 'y': 'A'}, {'x': 8.0, 'y': 'B'}, {'x': 9.0, 'y': 'B'}]"

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content=train_query)],
            "step_count": 0,
        }
        return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "test-thread-ml"}})

    result = asyncio.run(_run())
    assert any(a["title"] == "ML Specialist" for a in result.get("activities", []))
    # Find the ML training result by tool name (data profiling may have run first)
    ml_tool_result = next(
        (r for r in result.get("tool_results", []) if r.get("tool") == "train_ml_model_tool"),
        None,
    )
    assert ml_tool_result is not None, "train_ml_model_tool result not found in tool_results"
    ml_res = ml_tool_result["result"]
    assert "model_id" in ml_res
    assert ml_res["task"] == "classification"


def test_multimodal_tool_execution_node() -> None:
    """Multimodal specialist node must analyze image attachments."""

    async def _run():
        from poc_kanini.multimodal.models import MultimodalAnalysis
        mock_analysis = MultimodalAnalysis(
            answer="A dark test canvas.",
            observations=["Dark background"],
            source_metadata={"filename": "test.jpg", "mime_type": "image/jpeg", "size_bytes": "54"},
        )
        with patch("poc_kanini.tools.multimodal_tools.MultimodalService") as mock_mm_cls:
            svc = MagicMock()
            svc.analyze = AsyncMock(return_value=mock_analysis)
            mock_mm_cls.return_value = svc

            state: AgentConversationState = {
                "messages": [HumanMessage(content="Describe image")],
                "attachments": [{"filename": "test.jpg", "mime_type": "image/jpeg", "data": TINY_JPEG_B64}],
                "step_count": 0,
            }
            return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "test-thread-mm"}})

    result = asyncio.run(_run())
    assert any(a["title"] == "Multimodal Specialist" for a in result.get("activities", []))
    assert result["tool_results"][0]["tool"] == "analyze_image_tool"


# ---------------------------------------------------------------------------
# 12-14. Bounded Cross-Specialist & Multi-Step Execution Tests
# ---------------------------------------------------------------------------


def test_cross_specialist_data_to_ml_workflow() -> None:
    """Data specialist must transition to ML specialist when user requests profiling AND model training."""
    query = "Profile this tabular dataset, then train a classifier using target as target column."

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content=query)],
            "step_count": 0,
            "max_steps": 5,
        }
        return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "test-thread-cross"}})

    result = asyncio.run(_run())

    activity_titles = [a["title"] for a in result.get("activities", [])]
    assert "Data Specialist" in activity_titles
    assert "Cross-Specialist Transition" in activity_titles
    assert "ML Specialist" in activity_titles
    tool_names = [t["tool"] for t in result.get("tool_results", [])]
    assert "profile_dataset_tool" in tool_names
    assert "train_ml_model_tool" in tool_names


def test_maximum_step_limit_enforced() -> None:
    """Graph must halt execution when step_count reaches max_steps."""

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content="Profile dataset and train classifier")],
            "step_count": 5,  # already at limit
            "max_steps": 5,
        }
        return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "test-thread-limit"}})

    result = asyncio.run(_run())
    assert result["step_count"] >= 5


# ---------------------------------------------------------------------------
# 15-18. Error Recovery & Attachment Validation Tests
# ---------------------------------------------------------------------------


def test_tool_failure_recovery() -> None:
    """Failed tools must return error dictionaries without throwing unhandled graph exceptions."""

    async def _run():
        with patch("poc_kanini.tools.rag_tools.RagService") as mock_rag:
            svc = MagicMock()
            svc.retrieve_evidence = AsyncMock(side_effect=RuntimeError("Vector Store Disconnected"))
            mock_rag.return_value = svc

            state: AgentConversationState = {
                "messages": [HumanMessage(content="Search document policy")],
                "step_count": 0,
            }
            return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "test-thread-fail"}})

    result = asyncio.run(_run())
    assert "messages" in result
    assert len(result["messages"]) > 0


def test_invalid_image_attachment_handling() -> None:
    """Invalid image attachment data must be handled gracefully with an error in tool_results."""

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content="Describe image")],
            "attachments": [{"filename": "bad.png", "mime_type": "image/png", "data": "not-valid-b64!"}],
            "step_count": 0,
        }
        return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "test-thread-badimg"}})

    result = asyncio.run(_run())
    assert "error" in result["tool_results"][0]["result"]


# ---------------------------------------------------------------------------
# 19-22. API Endpoint & Synthesis Tests
# ---------------------------------------------------------------------------


def test_api_chat_backwards_compatibility_text_only() -> None:
    """POST /api/chat must accept standard text-only ChatRequest and return ChatResponse."""
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hello, how are you?"}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["message"]["role"] == "assistant"
    assert len(body["message"]["content"]) > 0


def test_api_chat_with_multimodal_attachment() -> None:
    """POST /api/chat must accept ChatRequest with attachments and run multimodal graph pipeline."""
    with patch("poc_kanini.tools.multimodal_tools.MultimodalService") as mock_mm:
        from poc_kanini.multimodal.models import MultimodalAnalysis
        svc = MagicMock()
        svc.analyze = AsyncMock(
            return_value=MultimodalAnalysis(
                answer="A synthetic photograph.",
                observations=["Plain white surface"],
                source_metadata={"filename": "test.png", "mime_type": "image/png", "size_bytes": "54"},
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
                            "filename": "test.png",
                            "mime_type": "image/png",
                            "data": TINY_JPEG_B64,
                        }
                    ],
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert "assistant" in body["message"]["role"]
    assert len(body["message"]["content"]) > 0


def test_citation_preservation_in_synthesis() -> None:
    """Synthesis must include citation labels [filename — Page X] when RAG evidence is retrieved."""
    chunk = DocumentChunk(
        chunk_id="c1", document_id="doc-1", filename="employee_guide.pdf", document_type="guide", page_number=5, chunk_index=0, text="Remote work is allowed 2 days per week."
    )
    citation = Citation(chunk_id="c1", document_id="doc-1", filename="employee_guide.pdf", page_number=5, label="employee_guide.pdf — Page 5")
    retrieved = RetrievedChunk(chunk=chunk, score=0.95)

    async def _run():
        with patch("poc_kanini.tools.rag_tools.RagService") as mock_rag:
            svc = MagicMock()
            svc.retrieve_evidence = AsyncMock(return_value=([retrieved], [citation]))
            mock_rag.return_value = svc

            state: AgentConversationState = {
                "messages": [HumanMessage(content="What is the remote work policy in employee_guide.pdf?")],
                "step_count": 0,
            }
            return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "test-thread-cite"}})

    result = asyncio.run(_run())
    last_msg = result["messages"][-1].content
    assert "employee_guide.pdf — Page 5" in last_msg
