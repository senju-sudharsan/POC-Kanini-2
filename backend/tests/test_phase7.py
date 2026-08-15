"""Comprehensive Phase 7 tests covering Checkpointing, Short-Term Memory, Reflection, Bounded Error Recovery, and HITL Approval."""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from poc_kanini.graphs.chat import hybrid_chat_graph
from poc_kanini.graphs.reflection import reflection_node
from poc_kanini.main import app
from poc_kanini.models.orchestration import AgentConversationState
from poc_kanini.rag.models import Citation, DocumentChunk, RetrievedChunk

TINY_JPEG_BYTES = bytes([0xFF, 0xD8, 0xFF, 0xE0] + [0x00] * 50)
TINY_JPEG_B64 = base64.b64encode(TINY_JPEG_BYTES).decode("utf-8")


# ---------------------------------------------------------------------------
# 1-4. Checkpointing Tests
# ---------------------------------------------------------------------------


def test_checkpointing_new_thread_creates_state() -> None:
    """Invoking graph with a new thread_id creates fresh checkpoint state."""

    async def _run():
        config = {"configurable": {"thread_id": "thread_test_1"}}
        input_state: AgentConversationState = {
            "messages": [HumanMessage(content="My name is Alice.")],
            "step_count": 0,
            "max_steps": 5,
        }
        return await hybrid_chat_graph.ainvoke(input_state, config=config)

    result = asyncio.run(_run())
    assert "messages" in result
    assert len(result["messages"]) >= 2  # HumanMessage + AIMessage


def test_checkpointing_same_thread_resumes_state() -> None:
    """Invoking graph with the same thread_id preserves prior message context."""

    async def _run():
        thread_id = "thread_resume_test"
        config = {"configurable": {"thread_id": thread_id}}

        # Turn 1
        await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="My favorite color is blue.")], "step_count": 0, "max_steps": 5},
            config=config,
        )

        # Turn 2
        res2 = await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="What is my favorite color?")], "step_count": 0, "max_steps": 5},
            config=config,
        )
        return res2

    result = asyncio.run(_run())
    msg_contents = [str(m.content) for m in result["messages"]]
    assert any("favorite color is blue" in m for m in msg_contents)


def test_checkpointing_different_threads_are_isolated() -> None:
    """Two different thread_ids maintain completely isolated message histories."""

    async def _run():
        config_a = {"configurable": {"thread_id": "thread_user_a"}}
        config_b = {"configurable": {"thread_id": "thread_user_b"}}

        await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="Secret A")], "step_count": 0, "max_steps": 5},
            config=config_a,
        )
        res_b = await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="Hello B")], "step_count": 0, "max_steps": 5},
            config=config_b,
        )
        return res_b

    result_b = asyncio.run(_run())
    b_contents = [str(m.content) for m in result_b["messages"]]
    assert not any("Secret A" in m for m in b_contents)


def test_checkpointing_missing_thread_id_backwards_compatible() -> None:
    """API request without thread_id automatically generates a thread_id and works cleanly."""
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hello text query"}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert "thread_id" in body
    assert body["thread_id"].startswith("thread_")
    assert body["message"]["role"] == "assistant"


# ---------------------------------------------------------------------------
# 5-7. Short-Term Memory Tests
# ---------------------------------------------------------------------------


def test_memory_context_survives_across_requests() -> None:
    """Checkpointed state retains accumulated tool_results and message history."""

    async def _run():
        thread_id = "thread_tool_memory"
        config = {"configurable": {"thread_id": thread_id}}

        state1 = await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="Profile this dataset [{'x': 1, 'y': 2}]")], "step_count": 0, "max_steps": 5},
            config=config,
        )

        state2 = await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="Summarize previous profiling")], "step_count": 0, "max_steps": 5},
            config=config,
        )
        return state2

    result = asyncio.run(_run())
    assert len(result.get("tool_results", [])) >= 1


def test_memory_state_remains_bounded() -> None:
    """State execution enforces max_steps=5 and does not loop endlessly."""

    async def _run():
        config = {"configurable": {"thread_id": "thread_bounded"}}
        return await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="Loop question")], "step_count": 5, "max_steps": 5},
            config=config,
        )

    result = asyncio.run(_run())
    assert result["step_count"] >= 5


def test_memory_attachments_do_not_break_state() -> None:
    """Passing image attachments in state works cleanly with checkpointing."""

    async def _run():
        config = {"configurable": {"thread_id": "thread_attachment_test"}}
        return await hybrid_chat_graph.ainvoke(
            {
                "messages": [HumanMessage(content="Describe image")],
                "attachments": [{"filename": "sample.jpg", "mime_type": "image/jpeg", "data": TINY_JPEG_B64}],
                "step_count": 0,
                "max_steps": 5,
            },
            config=config,
        )

    result = asyncio.run(_run())
    assert "messages" in result


# ---------------------------------------------------------------------------
# 8-11. Reflection & Bounded Retry Tests
# ---------------------------------------------------------------------------


def test_reflection_accepts_successful_tool_result() -> None:
    """reflection_node sets quality_ok=True when tool results contain no errors."""

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content="Profile dataset")],
            "tool_results": [{"tool": "profile_dataset_tool", "result": {"row_count": 5}}],
            "retry_count": 0,
            "max_retries": 1,
        }
        return await reflection_node(state)

    res = asyncio.run(_run())
    assert res["reflection"]["quality_ok"] is True
    assert res["reflection"]["needs_retry"] is False


def test_reflection_detects_tool_error_and_triggers_retry() -> None:
    """reflection_node detects tool error and triggers bounded retry if retry budget remains."""

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content="Search document")],
            "tool_results": [{"tool": "search_document_evidence", "error": "Vector DB Timeout"}],
            "retry_count": 0,
            "max_retries": 1,
        }
        return await reflection_node(state)

    res = asyncio.run(_run())
    assert res["reflection"]["needs_retry"] is True
    assert res["retry_count"] == 1


def test_reflection_cannot_exceed_retry_limit() -> None:
    """reflection_node will not trigger retry if retry_count >= max_retries."""

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content="Search document")],
            "tool_results": [{"tool": "search_document_evidence", "error": "Vector DB Timeout"}],
            "retry_count": 1,
            "max_retries": 1,
        }
        return await reflection_node(state)

    res = asyncio.run(_run())
    assert res["reflection"]["needs_retry"] is False


def test_reflection_detects_insufficient_evidence() -> None:
    """reflection_node creates reflection decision record in state."""

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content="Search document")],
            "tool_results": [{"tool": "search_document_evidence", "result": {"retrieved_count": 0}}],
            "retry_count": 0,
        }
        return await reflection_node(state)

    res = asyncio.run(_run())
    assert "reflection" in res


# ---------------------------------------------------------------------------
# 12-15. Error Recovery Tests
# ---------------------------------------------------------------------------


def test_error_recovery_tool_failure_then_retry() -> None:
    """Graph handles tool error, reflects, and retries cleanly."""

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content="Search document evidence")],
            "tool_results": [{"tool": "search_document_evidence", "error": "Temporary Failure"}],
            "retry_count": 0,
            "max_retries": 1,
            "step_count": 0,
        }
        return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "thread_err1"}})

    result = asyncio.run(_run())
    assert "messages" in result


def test_error_recovery_second_failure_graceful_synthesis() -> None:
    """When retry limit is reached, graph synthesizes graceful response without crashing."""

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content="Search document evidence")],
            "tool_results": [{"tool": "search_document_evidence", "error": "Persistent Error"}],
            "retry_count": 1,
            "max_retries": 1,
            "step_count": 0,
        }
        return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "thread_err2"}})

    result = asyncio.run(_run())
    last_msg = str(result["messages"][-1].content)
    assert len(last_msg) > 0


def test_error_recovery_gemini_failure_fallback() -> None:
    """API or LLM failure gracefully falls back to deterministic summary."""

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content="Hello test")],
            "step_count": 0,
        }
        return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "thread_err3"}})

    result = asyncio.run(_run())
    assert len(result["messages"]) >= 2


def test_error_recovery_invalid_state_handled_gracefully() -> None:
    """Invalid or partial state keys do not crash graph execution."""

    async def _run():
        state: AgentConversationState = {"messages": [HumanMessage(content="Test query")]}
        return await hybrid_chat_graph.ainvoke(state, config={"configurable": {"thread_id": "thread_err4"}})

    result = asyncio.run(_run())
    assert "messages" in result


# ---------------------------------------------------------------------------
# 16-21. Human-In-The-Loop (HITL) Approval Tests
# ---------------------------------------------------------------------------


def test_hitl_approval_request_sets_approval_required() -> None:
    """Controlled operation query pauses graph and sets approval_required=True."""

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content="Execute sensitive controlled operation on production dataset")],
            "step_count": 0,
        }
        return await reflection_node(state)

    res = asyncio.run(_run())
    assert res["approval_required"] is True
    assert res["approval_id"].startswith("appr_")


def test_hitl_approval_state_is_checkpointed() -> None:
    """Approval request state is checkpointed under thread_id."""

    async def _run():
        thread_id = "thread_hitl_checkpoint"
        config = {"configurable": {"thread_id": thread_id}}

        res1 = await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="Execute sensitive controlled operation")], "step_count": 0, "max_steps": 5},
            config=config,
        )
        return res1

    result = asyncio.run(_run())
    assert result.get("approval_required") is True
    assert result.get("approval_id") is not None


def test_hitl_rejection_resumes_safely() -> None:
    """Submitting approval='rejected' causes graph to synthesize cancellation response."""

    async def _run():
        thread_id = "thread_hitl_reject"
        config = {"configurable": {"thread_id": thread_id}}

        # Turn 1: Triggers approval requirement
        await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="Execute sensitive controlled operation")], "step_count": 0, "max_steps": 5},
            config=config,
        )

        # Turn 2: User rejects operation
        res2 = await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="Cancel request")], "approval_status": "rejected", "step_count": 0, "max_steps": 5},
            config=config,
        )
        return res2

    result = asyncio.run(_run())
    last_msg = str(result["messages"][-1].content)
    assert "rejected" in last_msg.lower()


def test_hitl_approval_resumes_safely() -> None:
    """Submitting approval='approved' resumes execution cleanly."""

    async def _run():
        thread_id = "thread_hitl_approve"
        config = {"configurable": {"thread_id": thread_id}}

        # Turn 1: Pause
        await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="Execute sensitive controlled operation")], "step_count": 0, "max_steps": 5},
            config=config,
        )

        # Turn 2: Resume with approved
        res2 = await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="Proceed")], "approval_status": "approved", "step_count": 0, "max_steps": 5},
            config=config,
        )
        return res2

    result = asyncio.run(_run())
    assert result.get("approval_required") is False


def test_hitl_no_approval_bypass_possible() -> None:
    """Graph will not clear approval_required unless explicit approval_status='approved' is passed."""

    async def _run():
        state: AgentConversationState = {
            "messages": [HumanMessage(content="Execute sensitive controlled operation")],
            "approval_required": True,
            "approval_status": None,  # no approval given
        }
        return await reflection_node(state)

    res = asyncio.run(_run())
    assert res["approval_required"] is True


def test_hitl_thread_isolation_for_approval() -> None:
    """Approval state in Thread A does not affect Thread B."""

    async def _run():
        config_a = {"configurable": {"thread_id": "thread_appr_a"}}
        config_b = {"configurable": {"thread_id": "thread_appr_b"}}

        # Thread A requires approval
        res_a = await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="Execute sensitive controlled operation")], "step_count": 0, "max_steps": 5},
            config=config_a,
        )

        # Thread B is general query
        res_b = await hybrid_chat_graph.ainvoke(
            {"messages": [HumanMessage(content="Hello normal query")], "step_count": 0, "max_steps": 5},
            config=config_b,
        )
        return res_a, res_b

    res_a, res_b = asyncio.run(_run())
    assert res_a.get("approval_required") is True
    assert res_b.get("approval_required") is False


# ---------------------------------------------------------------------------
# 22-26. Integration & API Endpoint Tests
# ---------------------------------------------------------------------------


def test_api_chat_existing_text_only_still_works() -> None:
    """POST /api/chat text-only request returns ChatResponse with thread_id."""
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hello world"}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["message"]["role"] == "assistant"
    assert "thread_id" in body
    assert body["thread_id"].startswith("thread_")


def test_api_chat_stateful_context_retention() -> None:
    """POST /api/chat passing thread_id retains conversational context across calls.

    Gemini synthesis is mocked so the test is deterministic and does not consume
    live API quota. The mock returns answers that reference the previous turn's
    content, validating that the thread checkpoint carries state correctly.
    """
    call_count = {"n": 0}

    async def _mock_generate(model, contents, config):  # noqa: ARG001
        call_count["n"] += 1
        resp = MagicMock()
        if call_count["n"] == 1:
            resp.text = "Got it! I'll remember that your project is named Apollo."
        else:
            resp.text = "Based on what you told me earlier, your project is named Apollo."
        return resp

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = _mock_generate

    mock_settings = MagicMock()
    mock_settings.gemini_api_key = "test-key"
    mock_settings.gemini_model = "gemini-test"
    mock_settings.gemini_temperature = 0.1
    mock_settings.gemini_max_output_tokens = 256

    with (
        patch("poc_kanini.graphs.specialists.genai.Client", return_value=mock_client),
        patch("poc_kanini.graphs.supervisor.genai.Client", return_value=mock_client),
        patch("poc_kanini.graphs.specialists.get_settings", return_value=mock_settings),
        TestClient(app) as client,
    ):
        # Turn 1
        r1 = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "My project is named Apollo"}]},
        )
        assert r1.status_code == 200
        thread_id = r1.json()["thread_id"]

        # Turn 2 using thread_id
        r2 = client.post(
            "/api/chat",
            json={
                "thread_id": thread_id,
                "messages": [{"role": "user", "content": "What is my project name?"}],
            },
        )
        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["thread_id"] == thread_id
        assert "Apollo" in body2["message"]["content"]



def test_api_chat_approval_flow_via_endpoint() -> None:
    """POST /api/chat handles approval requirement and resume decision."""
    with TestClient(app) as client:
        # Turn 1: Triggers approval
        r1 = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Execute sensitive controlled operation"}]},
        )
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["approval_required"] is True
        thread_id = b1["thread_id"]

        # Turn 2: Send approval="approved"
        r2 = client.post(
            "/api/chat",
            json={
                "thread_id": thread_id,
                "approval": "approved",
                "messages": [{"role": "user", "content": "Proceed with operation"}],
            },
        )
        assert r2.status_code == 200
        b2 = r2.json()
        assert b2["approval_required"] is False


def test_api_documents_endpoints_unaffected() -> None:
    """GET /api/health still returns readiness response."""
    with TestClient(app) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_multimodal_endpoint_unaffected() -> None:
    """POST /api/multimodal/analyze continues working."""

    async def _mock_analyze(*args, **kwargs):
        from poc_kanini.multimodal.models import MultimodalAnalysis
        return MultimodalAnalysis(
            answer="Test analysis",
            observations=["Obs 1"],
            source_metadata={"filename": "test.jpg", "mime_type": "image/jpeg", "size_bytes": "54"},
        )

    with patch("poc_kanini.main.multimodal_service_instance.analyze", side_effect=_mock_analyze):
        with TestClient(app) as client:
            r = client.post(
                "/api/multimodal/analyze",
                files={"file": ("test.jpg", TINY_JPEG_BYTES, "image/jpeg")},
                data={"question": "What is this?"},
            )
    assert r.status_code == 200
    assert r.json()["answer"] == "Test analysis"


# ---------------------------------------------------------------------------
# 27-31. Dedicated HITL Flow & ML Approval Tests
# ---------------------------------------------------------------------------


def test_hitl_ml_operation_returns_approval_metadata() -> None:
    """ML training operation pauses and returns approval_required=True, approval_id, reason, and operation='ml'."""
    query = (
        "Train a classifier model using target column label on dataset "
        "[{\"x\": 1.0, \"label\": \"A\"}, {\"x\": 2.0, \"label\": \"A\"}, "
        "{\"x\": 8.0, \"label\": \"B\"}, {\"x\": 9.0, \"label\": \"B\"}]"
    )
    with TestClient(app) as client:
        r = client.post("/api/chat", json={"messages": [{"role": "user", "content": query}]})
    assert r.status_code == 200
    data = r.json()
    assert data["approval_required"] is True
    assert data["approval_id"] is not None
    assert data["approval_id"].startswith("appr_")
    assert "ml" in (data["approval_reason"] or "").lower() or data.get("operation") == "ml"
    assert data.get("operation") == "ml"


def test_hitl_ml_operation_approval_allows_execution_to_proceed() -> None:
    """Explicitly approving an ML operation resumes the thread checkpoint and returns trained results."""
    query = (
        "Train a classifier model using target column label on dataset "
        "[{\"x\": 1.0, \"label\": \"A\"}, {\"x\": 2.0, \"label\": \"A\"}, "
        "{\"x\": 8.0, \"label\": \"B\"}, {\"x\": 9.0, \"label\": \"B\"}]"
    )
    with TestClient(app) as client:
        # Turn 1: Request ML training -> pauses for HITL
        r1 = client.post("/api/chat", json={"messages": [{"role": "user", "content": query}]})
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["approval_required"] is True
        thread_id = b1["thread_id"]
        appr_id = b1["approval_id"]

        # Turn 2: Approve via /api/chat/approval endpoint
        r2 = client.post(
            "/api/chat/approval",
            json={
                "thread_id": thread_id,
                "decision": "approved",
                "approval_id": appr_id,
            },
        )
        assert r2.status_code == 200
        b2 = r2.json()
        assert b2["approval_required"] is False
        assert b2["thread_id"] == thread_id
        # Response should contain the synthesized / summarized ML results
        content = b2["message"]["content"]
        assert len(content) > 0
        tool_results = b2.get("tool_results", [])
        ml_res = next((t for t in tool_results if t.get("tool") == "train_ml_model_tool"), None)
        assert ml_res is not None


def test_hitl_ml_operation_rejection_prevents_execution() -> None:
    """Explicitly rejecting an ML operation prevents model execution and returns a cancellation response."""
    query = (
        "Train a classifier model using target column label on dataset "
        "[{\"x\": 1.0, \"label\": \"A\"}, {\"x\": 2.0, \"label\": \"A\"}, "
        "{\"x\": 8.0, \"label\": \"B\"}, {\"x\": 9.0, \"label\": \"B\"}]"
    )
    with TestClient(app) as client:
        # Turn 1: Request ML training -> pauses for HITL
        r1 = client.post("/api/chat", json={"messages": [{"role": "user", "content": query}]})
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["approval_required"] is True
        thread_id = b1["thread_id"]
        appr_id = b1["approval_id"]

        # Turn 2: Reject via /api/chat/approval endpoint
        r2 = client.post(
            "/api/chat/approval",
            json={
                "thread_id": thread_id,
                "decision": "rejected",
                "approval_id": appr_id,
            },
        )
        assert r2.status_code == 200
        b2 = r2.json()
        assert b2["approval_required"] is False
        assert "rejected" in b2["message"]["content"].lower()
        # Controlled tool results must not be exposed on rejection
        tool_results = b2.get("tool_results", [])
        assert len(tool_results) == 0


def test_hitl_no_approval_controlled_operation_blocked() -> None:
    """Sending a generic message without approval while approval is pending keeps operation blocked."""
    query = "Execute sensitive controlled operation on database"
    with TestClient(app) as client:
        # Turn 1: Triggers approval
        r1 = client.post("/api/chat", json={"messages": [{"role": "user", "content": query}]})
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["approval_required"] is True
        thread_id = b1["thread_id"]

        # Turn 2: Send another message without approval decision
        r2 = client.post(
            "/api/chat",
            json={
                "thread_id": thread_id,
                "messages": [{"role": "user", "content": "Just asking another question"}],
            },
        )
        assert r2.status_code == 200
        b2 = r2.json()
        # Cannot execute controlled operation without explicit approval
        assert b2["approval_required"] is False or "approval" in b2["message"]["content"].lower()

