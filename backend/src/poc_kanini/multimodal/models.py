"""Pydantic models for multimodal image analysis requests and responses."""

from pydantic import BaseModel, Field


# Maximum allowed image upload size: 10 MB
MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024

# Supported MIME types for image upload
SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)

# Human-readable supported format descriptions (for error messages)
SUPPORTED_FORMAT_NAMES: str = "JPEG, PNG, WEBP"


class VisualObservation(BaseModel):
    """A discrete visual element or observation extracted from an image."""

    description: str = Field(
        ..., description="Description of the observed element or finding"
    )
    category: str = Field(
        default="general",
        description="Category of observation: e.g. 'text', 'chart', 'object', 'color', 'structure'",
    )


class MultimodalAnalysis(BaseModel):
    """Structured result from a Gemini multimodal image analysis request."""

    answer: str = Field(
        ..., description="Direct answer to the user's question about the image"
    )
    observations: list[str] = Field(
        default_factory=list,
        description="Key visual observations made about the image",
    )
    detected_elements: list[VisualObservation] = Field(
        default_factory=list,
        description="Specific visual elements identified in the image (text, charts, objects, etc.)",
    )
    uncertainty_notes: list[str] = Field(
        default_factory=list,
        description="Anything the model is uncertain about or cannot reliably determine from the image",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Validation warnings or processing notices",
    )
    source_metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Metadata about the source image (filename, mime_type, size_bytes)",
    )
