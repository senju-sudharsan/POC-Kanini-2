"""
Deterministic tests for the 6 cleanup items raised in the production review.

No real Gemini API calls are made. All provider boundaries are mocked.

Issue 1 — No-data mock dataset removal
Issue 2 — Datetime column classification
Issue 3 — Profiling results survive Gemini quota failure
Issue 4 — Multimodal report cleanup
Issue 5 — AURA request counter (backend state, not UI)
Issue 6 — ML capability offline routing
"""

import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage

from poc_kanini.graphs.specialists import data_agent_node, ml_agent_node
from poc_kanini.ml.models import DatasetProfile
from poc_kanini.ml.profiler import profile_dataframe
from poc_kanini.services.report_service import generate_report


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_state(**kwargs):
    return {"messages": [HumanMessage(content=kwargs.pop("query", "profile this dataset"))],
            "step_count": 0, "max_steps": 5, **kwargs}


# ═══════════════════════════════════════════════════════════════════════════
# ISSUE 1 — No-data mock dataset removal
# ═══════════════════════════════════════════════════════════════════════════

def test_data_agent_no_data_returns_honest_error_not_mock_rows() -> None:
    """data_agent_node must NOT fabricate feature1/feature2/churn rows.
    When no csv_data and no parsable inline data, tool_results must contain
    an 'error' key — not a 'result' key with fake row counts.
    """
    state = _make_state(query="Profile a sample dataset and show columns")

    async def _run():
        return await data_agent_node(state)

    result = asyncio.run(_run())
    tools = result.get("tool_results", [])
    profile_results = [t for t in tools if t.get("tool") == "profile_dataset_tool"]
    assert profile_results, "Expected a profile_dataset_tool entry"
    entry = profile_results[0]
    # Must have an error — not a successful result with fabricated rows
    assert "error" in entry, "Expected an error, got a result (mock data is still being generated)"
    assert "result" not in entry, "Mock result must not be present"
    assert "feature1" not in str(entry), "Fabricated column 'feature1' must not appear"
    assert "feature2" not in str(entry), "Fabricated column 'feature2' must not appear"
    assert "churn" not in str(entry), "Fabricated column 'churn' must not appear"


def test_data_agent_no_data_message_is_user_friendly() -> None:
    """The honest error message must be informative to the user."""
    state = _make_state(query="Show me dataset columns")

    async def _run():
        return await data_agent_node(state)

    result = asyncio.run(_run())
    tools = result.get("tool_results", [])
    entry = next((t for t in tools if t.get("tool") == "profile_dataset_tool"), None)
    assert entry is not None
    msg = entry.get("error", "").lower()
    # Should mention dataset / csv / file
    assert any(word in msg for word in ["dataset", "csv", "file", "data"]), (
        f"Error message not user-friendly: '{msg}'"
    )


def test_data_agent_with_csv_data_profiles_correctly() -> None:
    """csv_data from state MUST be profiled — no honest-error short-circuit."""
    csv = "name,age,score\nAlice,30,88.5\nBob,25,72.0\nCarol,35,95.3"
    state = _make_state(query="profile this", csv_data=csv)

    async def _run():
        return await data_agent_node(state)

    result = asyncio.run(_run())
    tools = result.get("tool_results", [])
    entry = next((t for t in tools if t.get("tool") == "profile_dataset_tool"), None)
    assert entry is not None, "profile_dataset_tool entry missing"
    assert "result" in entry, "Expected a successful result"
    assert entry["result"]["row_count"] == 3
    assert entry["result"]["column_count"] == 3
    # 'name' is categorical, 'age'/'score' are numeric
    assert "age" in entry["result"]["numeric_columns"] or "score" in entry["result"]["numeric_columns"]


# ═══════════════════════════════════════════════════════════════════════════
# ISSUE 2 — Datetime column classification
# ═══════════════════════════════════════════════════════════════════════════

def _build_mixed_df() -> pd.DataFrame:
    """Smallest generic DataFrame with numeric, categorical, and datetime columns."""
    return pd.DataFrame({
        "age": [25, 30, 35, None],                       # numeric
        "category": ["A", "B", "A", "C"],                # categorical
        "event_date": [                                   # datetime
            "2023-01-15 08:30:00",
            "2023-02-20 12:00:00",
            "2023-03-10 09:15:00",
            "2023-04-05 07:45:00",
        ],
    })


def test_profiler_detects_numeric_columns() -> None:
    df = _build_mixed_df()
    profile = profile_dataframe(df)
    assert "age" in profile.numeric_columns


def test_profiler_detects_categorical_columns() -> None:
    df = _build_mixed_df()
    profile = profile_dataframe(df)
    assert "category" in profile.categorical_columns
    assert "category" not in profile.datetime_columns


def test_profiler_detects_datetime_columns() -> None:
    df = _build_mixed_df()
    profile = profile_dataframe(df)
    assert "event_date" in profile.datetime_columns, (
        "Datetime column not classified — timestamps are being misclassified as categorical"
    )
    # Datetime columns must NOT appear in categorical
    assert "event_date" not in profile.categorical_columns, (
        "Datetime column must not also appear in categorical_columns"
    )


def test_profiler_datetime_not_in_numeric() -> None:
    df = _build_mixed_df()
    profile = profile_dataframe(df)
    assert "event_date" not in profile.numeric_columns


def test_profiler_handles_missing_values() -> None:
    df = _build_mixed_df()  # 'age' has one None
    profile = profile_dataframe(df)
    assert profile.missing_counts.get("age", 0) == 1


def test_profiler_dataset_profile_has_datetime_columns_field() -> None:
    """DatasetProfile model must expose datetime_columns list."""
    df = _build_mixed_df()
    profile = profile_dataframe(df)
    assert hasattr(profile, "datetime_columns"), "DatasetProfile missing 'datetime_columns' field"
    assert isinstance(profile.datetime_columns, list)


def test_profiler_timestamp_columns_named_generically() -> None:
    """Detection should work for any column name that contains date-like values."""
    df = pd.DataFrame({
        "order_ts": ["2024-01-01 00:00:00"] * 5,
        "delivery_estimated": ["2024-03-15"] * 5,
        "count": [1, 2, 3, 4, 5],
    })
    profile = profile_dataframe(df)
    assert "order_ts" in profile.datetime_columns
    assert "delivery_estimated" in profile.datetime_columns
    assert "count" in profile.numeric_columns


# ═══════════════════════════════════════════════════════════════════════════
# ISSUE 3 — Profiling results survive Gemini quota failure
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_includes_profiling_result_when_gemini_fails() -> None:
    """When Gemini synthesis raises a 429 quota error, the profiling summary
    must still appear in the response message — row_count, column_count, etc.
    """
    from fastapi.testclient import TestClient
    from poc_kanini.main import app

    csv = "city,population,founded\nChennai,7000000,1639-08-22\nMumbai,12500000,1661-05-23"

    class _QuotaError(Exception):
        pass

    async def _raise_quota(*args, **kwargs):
        raise _QuotaError("Quota exceeded: 429")

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = _raise_quota

    with (
        patch("poc_kanini.graphs.specialists.genai.Client", return_value=mock_client),
        patch("poc_kanini.graphs.supervisor.genai.Client", return_value=mock_client),
        TestClient(app) as client,
    ):
        r = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Profile this CSV dataset"}],
                "csv_data": csv,
            },
        )
    assert r.status_code == 200
    body = r.json()
    content = body["message"]["content"]
    # The deterministic fallback must mention row/column counts
    assert "2" in content or "population" in content or "city" in content, (
        f"Profiling result lost after Gemini quota failure. Got: {content!r}"
    )
    # Quota exhausted warning must be present
    assert body.get("synthesis_status") in ("quota_exhausted", "degraded", "error"), (
        f"Expected degraded status, got: {body.get('synthesis_status')!r}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# ISSUE 4 — Multimodal report cleanup
# ═══════════════════════════════════════════════════════════════════════════

def test_image_analysis_success_report_has_no_generic_recommendation() -> None:
    """Successful image analysis must NOT emit the generic 'Confirm visual
    observation metadata against primary documentation' recommendation.
    """
    tool_results = [
        {
            "tool": "analyze_image_tool",
            "result": {
                "answer": "Photo shows a coastal sunset with vivid orange sky.",
                "observations": ["Orange sky", "Sea horizon", "Silhouetted trees"],
                "source_metadata": {"filename": "sunset.jpg"},
            },
        }
    ]
    report = generate_report(
        report_type="image_analysis",
        tool_results=tool_results,
        user_query="describe this image",
    )
    rec_text = " ".join(report.recommendations).lower()
    assert "confirm visual observation metadata" not in rec_text, (
        "Generic boilerplate recommendation must be removed from successful image analysis"
    )


def test_image_analysis_failure_report_does_not_claim_completed() -> None:
    """Failed image analysis must NOT claim 'Visual analysis completed' — it
    must say analysis was not completed.
    """
    tool_results = [
        {
            "tool": "analyze_image_tool",
            "error": "API call failed: timeout",
        }
    ]
    report = generate_report(
        report_type="image_analysis",
        tool_results=tool_results,
        user_query="analyze this image",
    )
    sections = report.sections
    assert sections, "Expected at least one section"
    combined = " ".join(s.content for s in sections).lower()
    assert "not completed" in combined or "error" in combined, (
        f"Failed analysis must say 'not completed'. Got: {combined!r}"
    )
    # Must NOT fabricate positive claim
    assert "visual analysis completed" not in combined


def test_image_analysis_success_report_includes_answer() -> None:
    """Successful image analysis report must include the answer text."""
    tool_results = [
        {
            "tool": "analyze_image_tool",
            "result": {
                "answer": "A bar chart showing monthly revenue growth.",
                "observations": ["X-axis labeled 'Month'", "Y-axis labeled 'Revenue'"],
                "source_metadata": {"filename": "chart.png"},
            },
        }
    ]
    report = generate_report(
        report_type="image_analysis",
        tool_results=tool_results,
        user_query="describe this chart",
    )
    combined = " ".join(s.content for s in report.sections)
    assert "bar chart" in combined or "revenue" in combined.lower()


# ═══════════════════════════════════════════════════════════════════════════
# ISSUE 6 — ML capability offline testing
# ═══════════════════════════════════════════════════════════════════════════

def test_ml_agent_trains_model_from_inline_data_via_graph() -> None:
    """ML agent must: receive query with inline dataset → route to 'ml'
    → invoke train_ml_model_tool → return model_id and metrics.
    No Gemini calls made (synthesis is mocked).
    """
    from fastapi.testclient import TestClient
    from poc_kanini.main import app

    async def _mock_synth(*args, **kwargs):
        resp = MagicMock()
        resp.text = "Model trained successfully."
        return resp

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = _mock_synth

    query = (
        "Train a classifier model using target column label on dataset "
        "[{\"x\": 1.0, \"label\": \"A\"}, {\"x\": 2.0, \"label\": \"A\"}, "
        "{\"x\": 8.0, \"label\": \"B\"}, {\"x\": 9.0, \"label\": \"B\"}]"
    )

    with (
        patch("poc_kanini.graphs.specialists.genai.Client", return_value=mock_client),
        patch("poc_kanini.graphs.supervisor.genai.Client", return_value=mock_client),
        TestClient(app) as client,
    ):
        r = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": query}]},
        )

    assert r.status_code == 200
    body = r.json()
    tool_results = body.get("tool_results", [])
    ml_result = next(
        (t for t in tool_results if t.get("tool") == "train_ml_model_tool" and t.get("result")),
        None,
    )
    assert ml_result is not None, (
        f"train_ml_model_tool result not found. tool_results={tool_results}"
    )
    assert "model_id" in ml_result["result"]
    assert ml_result["result"].get("task") == "classification"


def test_ml_agent_no_data_returns_honest_error() -> None:
    """ml_agent_node must NOT fabricate feature1/feature2 rows when no data is provided."""
    state = _make_state(query="Train a classifier model")

    async def _run():
        return await ml_agent_node(state)

    result = asyncio.run(_run())
    tools = result.get("tool_results", [])
    ml_results = [t for t in tools if t.get("tool") == "train_ml_model_tool"]
    assert ml_results, "Expected a train_ml_model_tool entry"
    entry = ml_results[0]
    assert "error" in entry, "Expected an honest error when no dataset is provided"
    assert "result" not in entry, "Must not return a fake trained model"
    assert "feature1" not in str(entry)
    assert "feature2" not in str(entry)


def test_ml_predict_tool_uses_trained_model_id() -> None:
    """Prediction tool must accept the model_id from a prior training call."""
    from poc_kanini.tools.ml_tools import predict_ml_model_tool, train_ml_model_tool

    train_data = [
        {"x": 1.0, "y": 0},
        {"x": 2.0, "y": 0},
        {"x": 8.0, "y": 1},
        {"x": 9.0, "y": 1},
    ]
    train_result = train_ml_model_tool.invoke({"data": train_data, "target": "y"})
    assert "model_id" in train_result

    model_id = train_result["model_id"]
    pred_data = [{"x": 1.5}, {"x": 8.5}]
    pred_result = predict_ml_model_tool.invoke({"model_id": model_id, "data": pred_data})
    assert "predictions" in pred_result
    assert len(pred_result["predictions"]) == 2


def test_ml_train_tool_returns_metrics() -> None:
    """train_ml_model_tool must return classification metrics (accuracy/f1)."""
    from poc_kanini.tools.ml_tools import train_ml_model_tool

    data = [
        {"a": 1.0, "b": 2.0, "target": "yes"},
        {"a": 1.5, "b": 1.8, "target": "yes"},
        {"a": 5.0, "b": 8.0, "target": "no"},
        {"a": 5.5, "b": 8.2, "target": "no"},
    ]
    result = train_ml_model_tool.invoke({"data": data, "target": "target"})
    assert "metrics" in result
    metrics = result["metrics"]
    # Classification task → accuracy must be present
    assert "accuracy" in metrics
    assert metrics["accuracy"] is not None
