"""Tests for Phase 4B — Multimodal AI capability.

All tests are deterministic and do not require a live Gemini API key.
Gemini calls are fully mocked.
"""

from __future__ import annotations

import asyncio
import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from poc_kanini.multimodal.models import (
    MAX_IMAGE_SIZE_BYTES,
    SUPPORTED_MIME_TYPES,
    MultimodalAnalysis,
    VisualObservation,
)
from poc_kanini.multimodal.validator import ImageValidationError, validate_image


# ---------------------------------------------------------------------------
# Tiny synthetic images (valid headers only — not real image data, but the
# validator only checks MIME type and size, so these pass validation)
# ---------------------------------------------------------------------------

TINY_JPEG_BYTES = bytes(
    [0xFF, 0xD8, 0xFF, 0xE0] + [0x00] * 100
)  # JPEG magic + padding
TINY_PNG_BYTES = bytes(
    [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] + [0x00] * 100
)  # PNG magic
TINY_WEBP_BYTES = bytes(b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 100)  # WEBP


# ---------------------------------------------------------------------------
# 1. Supported image validation — valid inputs pass without exception
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content,mime_type",
    [
        (TINY_JPEG_BYTES, "image/jpeg"),
        (TINY_PNG_BYTES, "image/png"),
        (TINY_WEBP_BYTES, "image/webp"),
    ],
)
def test_validate_image_supported_formats(content: bytes, mime_type: str) -> None:
    """Valid image MIME types within the size limit must pass validation."""
    # Should not raise
    validate_image(content=content, mime_type=mime_type, filename="test_image")


# ---------------------------------------------------------------------------
# 2. Unsupported MIME type raises ImageValidationError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mime_type",
    ["application/pdf", "image/gif", "image/bmp", "text/plain", "video/mp4", ""],
)
def test_validate_image_unsupported_mime(mime_type: str) -> None:
    """Unsupported MIME types must raise ImageValidationError."""
    with pytest.raises(ImageValidationError, match="Unsupported image format"):
        validate_image(content=TINY_JPEG_BYTES, mime_type=mime_type, filename="bad.gif")


# ---------------------------------------------------------------------------
# 3. Empty input raises ImageValidationError
# ---------------------------------------------------------------------------


def test_validate_image_empty_bytes() -> None:
    """Empty bytes must raise ImageValidationError."""
    with pytest.raises(ImageValidationError, match="empty"):
        validate_image(content=b"", mime_type="image/jpeg", filename="empty.jpg")


# ---------------------------------------------------------------------------
# 4. Oversized input raises ImageValidationError
# ---------------------------------------------------------------------------


def test_validate_image_oversized() -> None:
    """Images exceeding MAX_IMAGE_SIZE_BYTES must raise ImageValidationError."""
    oversized = b"x" * (MAX_IMAGE_SIZE_BYTES + 1)
    with pytest.raises(ImageValidationError, match="exceeds"):
        validate_image(content=oversized, mime_type="image/jpeg", filename="big.jpg")


# ---------------------------------------------------------------------------
# 5. Structured response model validation
# ---------------------------------------------------------------------------


def test_multimodal_analysis_model_defaults() -> None:
    """MultimodalAnalysis must build with only a required answer field."""
    analysis = MultimodalAnalysis(answer="A simple test image.")
    assert analysis.answer == "A simple test image."
    assert analysis.observations == []
    assert analysis.detected_elements == []
    assert analysis.uncertainty_notes == []
    assert analysis.warnings == []
    assert analysis.source_metadata == {}


def test_multimodal_analysis_model_full() -> None:
    """MultimodalAnalysis must accept all fields and serialise correctly."""
    obs = VisualObservation(description="A bar chart with 3 series", category="chart")
    analysis = MultimodalAnalysis(
        answer="This is a bar chart comparing sales.",
        observations=["Three coloured bars per group"],
        detected_elements=[obs],
        uncertainty_notes=["Y-axis units are unclear"],
        warnings=["Low contrast"],
        source_metadata={"filename": "chart.png"},
    )
    dumped = analysis.model_dump()
    assert dumped["answer"] == "This is a bar chart comparing sales."
    assert dumped["detected_elements"][0]["category"] == "chart"


# ---------------------------------------------------------------------------
# 6. GeminiMultimodalService parses a valid JSON response from Gemini
# ---------------------------------------------------------------------------


def test_gemini_service_parses_json_response() -> None:
    """GeminiMultimodalService must parse a Gemini JSON response correctly."""
    from poc_kanini.multimodal.gemini_service import GeminiMultimodalService
    from poc_kanini.core.config import Settings

    fake_response_data = {
        "answer": "This is a bar chart.",
        "observations": ["Three bars visible"],
        "detected_elements": [{"description": "Y-axis label 'Revenue'", "category": "text"}],
        "uncertainty_notes": ["Chart title is partially cut off"],
    }
    fake_response = MagicMock()
    fake_response.text = json.dumps(fake_response_data)

    settings = Settings(gemini_api_key="fake-key-for-test")
    service = GeminiMultimodalService(settings)

    async def _run() -> MultimodalAnalysis:
        with patch("poc_kanini.multimodal.gemini_service.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(return_value=fake_response)
            return await service.analyze_image(
                content=TINY_PNG_BYTES,
                mime_type="image/png",
                question="What is in this chart?",
                filename="chart.png",
            )

    result = asyncio.run(_run())
    assert result.answer == "This is a bar chart."
    assert "Three bars visible" in result.observations
    assert result.detected_elements[0].category == "text"
    assert "partially cut off" in result.uncertainty_notes[0]
    assert result.source_metadata["filename"] == "chart.png"


# ---------------------------------------------------------------------------
# 7. GeminiMultimodalService falls back gracefully when JSON is malformed
# ---------------------------------------------------------------------------


def test_gemini_service_fallback_on_bad_json() -> None:
    """Service must fall back to raw text if Gemini returns non-JSON output."""
    from poc_kanini.multimodal.gemini_service import GeminiMultimodalService
    from poc_kanini.core.config import Settings

    fake_response = MagicMock()
    fake_response.text = "This is a plain text answer without JSON."

    settings = Settings(gemini_api_key="fake-key")
    service = GeminiMultimodalService(settings)

    async def _run() -> MultimodalAnalysis:
        with patch("poc_kanini.multimodal.gemini_service.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(return_value=fake_response)
            return await service.analyze_image(
                content=TINY_JPEG_BYTES,
                mime_type="image/jpeg",
                question="What do you see?",
            )

    result = asyncio.run(_run())
    assert "plain text answer" in result.answer
    assert any("JSON" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 8. MultimodalService validates before calling Gemini
# ---------------------------------------------------------------------------


def test_multimodal_service_rejects_invalid_input() -> None:
    """MultimodalService must raise ImageValidationError before Gemini is called."""
    from poc_kanini.multimodal.service import MultimodalService
    from poc_kanini.core.config import Settings

    settings = Settings(gemini_api_key="fake-key")
    service = MultimodalService(settings)

    async def _run() -> None:
        with patch.object(service._gemini, "analyze_image", new_callable=AsyncMock) as mock_analyze:
            with pytest.raises(ImageValidationError):
                await service.analyze(
                    content=b"",  # empty — must fail validation
                    mime_type="image/jpeg",
                    question="What is this?",
                )
            mock_analyze.assert_not_called()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 9. API endpoint — validation error returns 400
# ---------------------------------------------------------------------------


def test_api_multimodal_analyze_unsupported_mime() -> None:
    """POST /api/multimodal/analyze must return 400 for an unsupported MIME type."""
    from poc_kanini.main import app

    with TestClient(app) as client:
        response = client.post(
            "/api/multimodal/analyze",
            files={"file": ("image.gif", TINY_JPEG_BYTES, "image/gif")},
            data={"question": "What is this?"},
        )
    assert response.status_code == 400
    assert "Unsupported image format" in response.json()["detail"]


def test_api_multimodal_analyze_empty_file() -> None:
    """POST /api/multimodal/analyze must return 400 for an empty file."""
    from poc_kanini.main import app

    with TestClient(app) as client:
        response = client.post(
            "/api/multimodal/analyze",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
            data={"question": "What is this?"},
        )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 9b. API endpoint — mocked successful Gemini response
# ---------------------------------------------------------------------------


def test_api_multimodal_analyze_mocked_success() -> None:
    """POST /api/multimodal/analyze returns structured response with a mocked Gemini call."""
    from poc_kanini.main import app

    mock_result = MultimodalAnalysis(
        answer="A photograph of a city skyline at night.",
        observations=["Tall buildings with illuminated windows", "Dark sky background"],
        detected_elements=[
            VisualObservation(description="Skyscraper cluster", category="object")
        ],
        source_metadata={"filename": "city.jpg", "mime_type": "image/jpeg", "size_bytes": str(len(TINY_JPEG_BYTES))},
    )

    with patch(
        "poc_kanini.main.multimodal_service_instance.analyze",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        with TestClient(app) as client:
            response = client.post(
                "/api/multimodal/analyze",
                files={"file": ("city.jpg", TINY_JPEG_BYTES, "image/jpeg")},
                data={"question": "Describe the image."},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A photograph of a city skyline at night."
    assert len(body["observations"]) == 2
    assert body["detected_elements"][0]["category"] == "object"
