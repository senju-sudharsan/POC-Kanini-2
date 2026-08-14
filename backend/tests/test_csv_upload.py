"""
Tests for CSV dataset upload and profiling via /api/chat.

All tests are deterministic -- Gemini provider calls are fully mocked so no
live API quota is consumed.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from poc_kanini.main import app
from poc_kanini.models.chat import ChatRequest
from poc_kanini.models.orchestration import AgentConversationState


SMALL_CSV = "order_id,status,price\n1,delivered,50.0\n2,shipped,30.0\n3,cancelled,0.0\n"


def test_chat_request_accepts_csv_data() -> None:
    req = ChatRequest(
        messages=[{"role": "user", "content": "Profile this dataset."}],
        csv_data=SMALL_CSV,
    )
    assert req.csv_data == SMALL_CSV


def test_chat_request_csv_data_defaults_to_none() -> None:
    req = ChatRequest(messages=[{"role": "user", "content": "Hello"}])
    assert req.csv_data is None


def test_agent_conversation_state_has_csv_data_field() -> None:
    state: AgentConversationState = {  # type: ignore[typeddict-item]
        "messages": [],
        "csv_data": SMALL_CSV,
    }
    assert state["csv_data"] == SMALL_CSV


@pytest.mark.anyio
async def test_data_agent_node_uses_csv_data_from_state() -> None:
    from poc_kanini.graphs.specialists import data_agent_node

    profile_result = {
        "row_count": 3,
        "column_count": 3,
        "columns": ["order_id", "status", "price"],
        "dtypes": {},
        "missing_values": {},
        "statistics": {},
    }
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = profile_result

    state: AgentConversationState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="Profile this dataset.")],
        "csv_data": SMALL_CSV,
        "step_count": 0,
        "max_steps": 5,
        "activities": [],
        "tool_results": [],
        "route": "data",
    }

    with patch("poc_kanini.graphs.specialists.profile_dataset_tool", mock_tool):
        result = await data_agent_node(state)

    mock_tool.invoke.assert_called_once_with({"data": SMALL_CSV})
    tool_results = result["tool_results"]
    assert any(tr.get("tool") == "profile_dataset_tool" for tr in tool_results)
    profiled = next(tr for tr in tool_results if tr.get("tool") == "profile_dataset_tool")
    assert profiled.get("result", {}).get("row_count") == 3


@pytest.mark.anyio
async def test_data_agent_node_no_data_returns_honest_error() -> None:
    from poc_kanini.graphs.specialists import data_agent_node

    state: AgentConversationState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="Profile a dataset.")],
        "csv_data": None,
        "step_count": 0,
        "max_steps": 5,
        "activities": [],
        "tool_results": [],
        "route": "data",
    }

    result = await data_agent_node(state)
    tool_results = result["tool_results"]
    assert tool_results, "Expected at least one tool_result entry"
    assert any("error" in tr for tr in tool_results), (
        "Expected error entry when no data provided, got: " + json.dumps(tool_results)
    )
    error_entry = next(tr for tr in tool_results if "error" in tr)
    error_text = error_entry["error"].lower()
    assert any(kw in error_text for kw in ("no dataset", "not provided", "please attach", "attach a csv")), (
        f"Error message not descriptive: {error_entry['error']}"
    )


def test_api_chat_csv_dataset_profiling_routes_to_data() -> None:
    client = TestClient(app, raise_server_exceptions=True)

    mock_profile_result = {
        "row_count": 3,
        "column_count": 3,
        "columns": ["order_id", "status", "price"],
        "dtypes": {"order_id": "int64", "status": "object", "price": "float64"},
        "missing_values": {"order_id": 0, "status": 0, "price": 0},
        "statistics": {"price": {"mean": 26.67, "std": 25.17, "min": 0.0, "max": 50.0}},
    }
    mock_profile = MagicMock()
    mock_profile.invoke.return_value = mock_profile_result

    mock_route_decision = MagicMock()
    mock_route_decision.route = "data"
    mock_route_decision.reason = "CSV data attached"
    mock_route_decision.confidence = 1.0
    mock_supervisor = AsyncMock(return_value=mock_route_decision)

    mock_synthesis_response = MagicMock()
    mock_synthesis_response.text = (
        "This dataset has 3 rows and 3 columns: order_id, status, price. "
        "No missing values detected."
    )

    with (
        patch("poc_kanini.graphs.specialists.profile_dataset_tool", mock_profile),
        patch("poc_kanini.graphs.supervisor.SupervisorRouter.route", mock_supervisor),
        patch(
            "poc_kanini.graphs.specialists.genai.Client",
            return_value=MagicMock(
                aio=MagicMock(
                    models=MagicMock(
                        generate_content=AsyncMock(return_value=mock_synthesis_response)
                    )
                )
            ),
        ),
    ):
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Profile this dataset."}],
                "csv_data": SMALL_CSV,
            },
        )

    assert response.status_code == 200, f"status={response.status_code} body={response.text}"
    body = response.json()
    assert "tool_results" in body
    profile_results = [tr for tr in body["tool_results"] if tr.get("tool") == "profile_dataset_tool"]
    assert profile_results, f"Expected profile_dataset_tool in tool_results, got: {body['tool_results']}"
    assert profile_results[0].get("result", {}).get("row_count") == 3


def test_api_chat_without_csv_data_still_works() -> None:
    client = TestClient(app, raise_server_exceptions=True)

    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["message"]["role"] == "assistant"

