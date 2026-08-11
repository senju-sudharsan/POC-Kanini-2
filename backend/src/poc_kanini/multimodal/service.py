"""MultimodalService — orchestrates validation and Gemini multimodal analysis."""

from poc_kanini.core.config import Settings
from poc_kanini.multimodal.gemini_service import GeminiMultimodalService
from poc_kanini.multimodal.models import MultimodalAnalysis
from poc_kanini.multimodal.validator import ImageValidationError, validate_image


class MultimodalService:
    """Validates image uploads and dispatches them to Gemini for analysis.

    Designed to be used as a FastAPI dependency and later wrapped as a
    LangGraph tool. The GeminiMultimodalService dependency can be replaced
    with a mock in tests.
    """

    def __init__(self, settings: Settings) -> None:
        self._gemini = GeminiMultimodalService(settings)

    async def analyze(
        self,
        content: bytes,
        mime_type: str,
        question: str,
        filename: str = "upload",
    ) -> MultimodalAnalysis:
        """Validate then analyze an image with Gemini.

        Args:
            content: Raw image bytes.
            mime_type: Declared MIME type of the image.
            question: The user's question about the image.
            filename: Original filename for error messages and metadata.

        Returns:
            A MultimodalAnalysis with structured visual observations.

        Raises:
            ImageValidationError: If the image fails validation.
            RuntimeError: If Gemini analysis fails.
        """
        validate_image(content=content, mime_type=mime_type, filename=filename)
        return await self._gemini.analyze_image(
            content=content,
            mime_type=mime_type,
            question=question,
            filename=filename,
        )
