"""Multimodal visual analysis tool wrapping MultimodalService."""

import base64
import binascii
import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from poc_kanini.core.config import get_settings
from poc_kanini.multimodal.service import MultimodalService
from poc_kanini.multimodal.validator import ImageValidationError

logger = logging.getLogger(__name__)


class MultimodalAnalyzeInput(BaseModel):
    """Input schema for multimodal image analysis tool."""

    image_base64: str = Field(
        ...,
        description=(
            "Base64-encoded image string (JPEG, PNG, or WEBP). "
            "Arbitrary filesystem paths are NOT accepted for security."
        ),
    )
    mime_type: str = Field(
        ...,
        description="Declared MIME type of the image (e.g. 'image/jpeg', 'image/png', 'image/webp').",
    )
    question: str = Field(
        default="Describe what you see in this image.",
        description="Specific question or analysis prompt for the visual input.",
    )
    filename: str = Field(
        default="upload.png",
        description="Optional filename for tracking.",
    )


@tool("analyze_image_tool", args_schema=MultimodalAnalyzeInput)
async def analyze_image_tool(
    image_base64: str,
    mime_type: str,
    question: str = "Describe what you see in this image.",
    filename: str = "upload.png",
) -> dict[str, Any]:
    """Analyse an image using Gemini multimodal understanding and return structured visual observations.

    Accepts base64-encoded image data and returns visual observations, category-tagged
    detected elements, explicit uncertainty notes, and an answer to the question.
    """
    try:
        content = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        return {"error": f"Invalid base64 encoding: {error}"}

    settings = get_settings()
    service = MultimodalService(settings)

    try:
        result = await service.analyze(
            content=content,
            mime_type=mime_type,
            question=question,
            filename=filename,
        )
        return result.model_dump()
    except ImageValidationError as error:
        return {"error": f"Validation error: {error}"}
    except RuntimeError as error:
        return {"error": str(error)}
    except Exception as error:
        logger.error("Error in analyze_image_tool: %s", error)
        return {"error": f"Image analysis failed: {error}"}
