"""Image validation logic for multimodal uploads."""

from poc_kanini.multimodal.models import (
    MAX_IMAGE_SIZE_BYTES,
    SUPPORTED_FORMAT_NAMES,
    SUPPORTED_MIME_TYPES,
)


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation checks."""
    pass


def validate_image(
    content: bytes,
    mime_type: str,
    filename: str = "upload",
) -> None:
    """Validate image content before forwarding to Gemini.

    Args:
        content: Raw image bytes.
        mime_type: MIME type string declared by the client.
        filename: Original filename (used for error messages only).

    Raises:
        ImageValidationError: If any validation check fails.
    """
    # 1. Empty content check
    if not content:
        raise ImageValidationError(
            f"Image '{filename}' is empty. Please upload a valid image file."
        )

    # 2. MIME type whitelist check
    # Normalize MIME type (strip parameters such as "; charset=utf-8")
    normalized_mime = (mime_type or "").split(";")[0].strip().lower()
    if normalized_mime not in SUPPORTED_MIME_TYPES:
        raise ImageValidationError(
            f"Unsupported image format '{normalized_mime}'. "
            f"Supported formats are: {SUPPORTED_FORMAT_NAMES}."
        )

    # 3. Size check
    size_bytes = len(content)
    if size_bytes > MAX_IMAGE_SIZE_BYTES:
        limit_mb = MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
        actual_mb = size_bytes / (1024 * 1024)
        raise ImageValidationError(
            f"Image '{filename}' is {actual_mb:.1f} MB, which exceeds the {limit_mb} MB limit."
        )
