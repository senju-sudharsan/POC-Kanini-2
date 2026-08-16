"""Tests for Phase 5 tool layer.

All tests are deterministic and do not require a live Gemini API key.
Gemini calls and RAG retrieval are mocked where needed.
"""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from poc_kanini.rag.models import Citation, DocumentChunk, RetrievedChunk
from poc_kanini.tools import (
    ALL_TOOLS,
    TOOLS_BY_NAME,
    analyze_image_tool,
    predict_ml_model_tool,
    profile_dataset_tool,
    search_document_evidence,
    train_ml_model_tool,
)


# ---------------------------------------------------------------------------
# 1. Tool Registry Tests
# ---------------------------------------------------------------------------


def test_tool_registry_exports() -> None:
    """Registry must export the planned tools with names & descriptions."""
    assert len(ALL_TOOLS) == 6
    expected_names = {
        "search_document_evidence",
        "profile_dataset_tool",
        "visualize_dataset_tool",
        "train_ml_model_tool",
        "predict_ml_model_tool",
        "analyze_image_tool",
    }
    assert set(TOOLS_BY_NAME.keys()) == expected_names
    for name, tool in TOOLS_BY_NAME.items():
        assert tool.name == name
        assert tool.description is not None and len(tool.description) > 10
        assert tool.args_schema is not None


# ---------------------------------------------------------------------------
# 2. RAG Tool Tests (search_document_evidence)
# ---------------------------------------------------------------------------


def test_search_document_evidence_valid() -> None:
    """search_document_evidence tool must return evidence snippets, citations, and summary without LLM call."""
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="doc-100",
        filename="leave_policy.pdf",
        document_type="policy",
        page_number=3,
        chunk_index=0,
        text="Employees are entitled to 20 days of paid annual leave.",
    )
    citation = Citation(
        chunk_id="c1",
        document_id="doc-100",
        filename="leave_policy.pdf",
        page_number=3,
        label="leave_policy.pdf — Page 3",
    )
    retrieved_source = RetrievedChunk(chunk=chunk, score=0.92, distance=0.08)

    async def _run():
        with patch("poc_kanini.tools.rag_tools.RagService") as mock_rag_cls:
            mock_service = MagicMock()
            mock_service.retrieve_evidence = AsyncMock(
                return_value=([retrieved_source], [citation])
            )
            mock_rag_cls.return_value = mock_service

            return await search_document_evidence.ainvoke(
                {"question": "How many leave days do employees get?", "document_id": "doc-100"}
            )

    result = asyncio.run(_run())

    assert result["retrieved_count"] == 1
    assert "Retrieved 1 evidence snippet" in result["summary"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["label"] == "leave_policy.pdf — Page 3"
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["text"] == "Employees are entitled to 20 days of paid annual leave."
    assert result["evidence"][0]["filename"] == "leave_policy.pdf"
    assert result["evidence"][0]["page_number"] == 3


def test_search_document_evidence_error_handling() -> None:
    """RAG tool must catch exceptions and return structured error response."""

    async def _run():
        with patch("poc_kanini.tools.rag_tools.RagService") as mock_rag_cls:
            mock_service = MagicMock()
            mock_service.retrieve_evidence = AsyncMock(side_effect=RuntimeError("Chroma DB connection error"))
            mock_rag_cls.return_value = mock_service

            return await search_document_evidence.ainvoke({"question": "What is the policy?"})

    result = asyncio.run(_run())

    assert "error" in result
    assert "Chroma DB connection error" in result["error"]
    assert result["evidence"] == []


# ---------------------------------------------------------------------------
# 3. Data Profiling Tool Tests (profile_dataset_tool)
# ---------------------------------------------------------------------------


def test_profile_dataset_tool_valid_dict_records() -> None:
    """profile_dataset_tool must profile a valid list of record dicts."""
    sample_data = [
        {"age": 25, "income": 50000.0, "dept": "Sales"},
        {"age": 30, "income": 60000.0, "dept": "Engineering"},
        {"age": 35, "income": 75000.0, "dept": "Sales"},
    ]
    result = profile_dataset_tool.invoke({"data": sample_data})

    assert "error" not in result
    assert result["row_count"] == 3
    assert result["column_count"] == 3
    assert set(result["columns"]) == {"age", "income", "dept"}
    assert "dept" in result["categorical_columns"]
    assert "age" in result["numeric_columns"]


def test_profile_dataset_tool_valid_csv_string() -> None:
    """profile_dataset_tool must profile a raw CSV string."""
    csv_data = "x,y\n1.0,2.0\n3.0,4.0\n"
    result = profile_dataset_tool.invoke({"data": csv_data})

    assert "error" not in result
    assert result["row_count"] == 2
    assert result["column_count"] == 2


def test_profile_dataset_tool_empty_data() -> None:
    """profile_dataset_tool must return an error dict for empty data."""
    result = profile_dataset_tool.invoke({"data": []})

    assert "error" in result
    assert "empty" in result["error"].lower()


# ---------------------------------------------------------------------------
# 4. ML Tools Tests (train_ml_model_tool & predict_ml_model_tool)
# ---------------------------------------------------------------------------


def test_ml_tools_train_and_predict_flow() -> None:
    """train_ml_model_tool must train a model, return model_id, and predict_ml_model_tool must use it."""
    train_data = [
        {"feature1": 1.0, "feature2": 2.0, "label": "A"},
        {"feature1": 1.5, "feature2": 1.8, "label": "A"},
        {"feature1": 5.0, "feature2": 8.0, "label": "B"},
        {"feature1": 5.5, "feature2": 8.2, "label": "B"},
    ]

    # 1. Train classification model
    train_result = train_ml_model_tool.invoke(
        {
            "data": train_data,
            "target": "label",
            "task": "classification",
            "model_type": "logistic",
        }
    )

    assert "error" not in train_result
    assert "model_id" in train_result
    model_id = train_result["model_id"]
    assert train_result["task"] == "classification"
    assert "metrics" in train_result
    assert "accuracy" in train_result["metrics"]
    assert len(train_result["feature_importance"]) == 2

    # 2. Predict using returned model_id
    test_data = [
        {"feature1": 1.1, "feature2": 1.9},
        {"feature1": 5.2, "feature2": 8.1},
    ]
    predict_result = predict_ml_model_tool.invoke(
        {"model_id": model_id, "data": test_data}
    )

    assert "error" not in predict_result
    assert predict_result["model_id"] == model_id
    assert len(predict_result["predictions"]) == 2
    assert predict_result["predictions"][0] in ["A", "B"]


def test_train_ml_model_tool_invalid_target() -> None:
    """train_ml_model_tool must return error dict for a non-existent target column."""
    train_data = [{"x": 1, "y": 2}]
    result = train_ml_model_tool.invoke({"data": train_data, "target": "non_existent_col"})

    assert "error" in result
    assert "non_existent_col" in result["error"]


def test_predict_ml_model_tool_invalid_model_id() -> None:
    """predict_ml_model_tool must return error dict for an uncached model_id."""
    result = predict_ml_model_tool.invoke(
        {"model_id": "non-existent-uuid-1234", "data": [{"x": 1}]}
    )

    assert "error" in result
    assert "non-existent-uuid-1234" in result["error"]


# ---------------------------------------------------------------------------
# 5. Multimodal Tool Tests (analyze_image_tool)
# ---------------------------------------------------------------------------


def test_analyze_image_tool_valid() -> None:
    """analyze_image_tool must accept base64 image data and return visual analysis."""
    # Synthetic tiny JPEG header bytes encoded as base64
    tiny_jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0] + [0x00] * 50)
    b64_image = base64.b64encode(tiny_jpeg).decode("utf-8")

    from poc_kanini.multimodal.models import MultimodalAnalysis, VisualObservation

    mock_analysis = MultimodalAnalysis(
        answer="A small synthetic test image.",
        observations=["Solid dark color pattern"],
        detected_elements=[VisualObservation(description="Canvas", category="structure")],
        uncertainty_notes=["No distinct objects"],
        source_metadata={"filename": "test.jpg", "mime_type": "image/jpeg", "size_bytes": str(len(tiny_jpeg))},
    )

    async def _run():
        with patch("poc_kanini.tools.multimodal_tools.MultimodalService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.analyze = AsyncMock(return_value=mock_analysis)
            mock_service_cls.return_value = mock_service

            return await analyze_image_tool.ainvoke(
                {
                    "image_base64": b64_image,
                    "mime_type": "image/jpeg",
                    "question": "What is in this image?",
                    "filename": "test.jpg",
                }
            )

    result = asyncio.run(_run())

    assert result["answer"] == "A small synthetic test image."
    assert "Solid dark color pattern" in result["observations"]
    assert result["detected_elements"][0]["category"] == "structure"
    assert result["source_metadata"]["filename"] == "test.jpg"


def test_analyze_image_tool_invalid_base64() -> None:
    """analyze_image_tool must return error dict for malformed base64 input."""

    async def _run():
        return await analyze_image_tool.ainvoke(
            {
                "image_base64": "!!!not_valid_base64!!!",
                "mime_type": "image/jpeg",
            }
        )

    result = asyncio.run(_run())

    assert "error" in result
    assert "base64" in result["error"].lower()


def test_analyze_image_tool_unsupported_mime() -> None:
    """analyze_image_tool must return error dict for unsupported MIME type."""
    tiny_pdf = b"%PDF-1.4 test"
    b64_pdf = base64.b64encode(tiny_pdf).decode("utf-8")

    async def _run():
        return await analyze_image_tool.ainvoke(
            {
                "image_base64": b64_pdf,
                "mime_type": "application/pdf",  # unsupported in multimodal tool
            }
        )

    result = asyncio.run(_run())

    assert "error" in result
    assert "Unsupported image format" in result["error"]
