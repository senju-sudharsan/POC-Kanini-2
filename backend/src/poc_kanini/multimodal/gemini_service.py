"""Gemini multimodal generation service for image understanding."""

import json
import re

from google import genai
from google.genai import types

from poc_kanini.core.config import Settings
from poc_kanini.multimodal.models import MultimodalAnalysis, VisualObservation


VISUAL_SYSTEM_INSTRUCTION = """You are the visual analysis specialist within AURA (Agentic Understanding & Retrieval Assistant).
Your task is to analyse images accurately and answer the user's question.

Follow these rules:
- Answer the user's question directly and concisely.
- Clearly distinguish between what you directly observe and what you infer.
- List specific visual elements you can see (text, charts, objects, colours, structure).
- Do NOT hallucinate details that are not visible in the image.
- When uncertain, explicitly state your uncertainty rather than guessing.
- Describe relevant visual evidence that supports your answer.
- Be concise unless the user requests detailed analysis.
- If the image contains text, quote it accurately.
- If the image contains charts or graphs, describe the type, axes, and key data points.

Output format (respond ONLY with valid JSON, no markdown fences):
{
  "answer": "<direct answer to the user question>",
  "observations": ["<observation 1>", "<observation 2>"],
  "detected_elements": [
    {"description": "<element description>", "category": "<text|chart|object|color|structure|other>"}
  ],
  "uncertainty_notes": ["<anything you are uncertain about>"]
}"""


class GeminiMultimodalService:
    """Send image content to Gemini and return structured visual analysis.

    Reuses the existing Gemini client pattern from services/gemini.py.
    The configured gemini_model (gemini-2.5-flash) supports multimodal input.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def analyze_image(
        self,
        content: bytes,
        mime_type: str,
        question: str,
        filename: str = "image",
    ) -> MultimodalAnalysis:
        """Send an image to Gemini with a question and return structured analysis.

        Args:
            content: Raw image bytes.
            mime_type: MIME type of the image (e.g. 'image/jpeg').
            question: The user's question about the image.
            filename: Original filename for metadata tracking.

        Returns:
            A MultimodalAnalysis with structured visual observations and answer.

        Raises:
            RuntimeError: If the API key is missing or Gemini call fails.
        """
        if not self._settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Set it in the .env file to enable multimodal analysis."
            )

        client = genai.Client(api_key=self._settings.gemini_api_key)

        # Build multimodal content: image bytes + user question text
        image_part = types.Part.from_bytes(data=content, mime_type=mime_type)
        question_part = types.Part.from_text(
            text=f"Question: {question.strip()}\n\nRespond only with JSON as instructed."
        )

        response = await client.aio.models.generate_content(
            model=self._settings.gemini_model,
            contents=[
                types.Content(
                    role="user",
                    parts=[image_part, question_part],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=VISUAL_SYSTEM_INSTRUCTION,
                temperature=0.1,  # Low temperature for factual visual description
                max_output_tokens=self._settings.gemini_max_output_tokens,
            ),
        )

        raw_text = (response.text or "").strip()
        return self._parse_response(
            raw_text=raw_text,
            filename=filename,
            mime_type=mime_type,
            size_bytes=len(content),
        )

    def _parse_response(
        self,
        raw_text: str,
        filename: str,
        mime_type: str,
        size_bytes: int,
    ) -> MultimodalAnalysis:
        """Parse the Gemini JSON response into a MultimodalAnalysis model.

        Gracefully falls back to a plain-text answer if JSON parsing fails,
        so a valid structured response is always returned.
        """
        warnings: list[str] = []
        source_metadata = {
            "filename": filename,
            "mime_type": mime_type,
            "size_bytes": str(size_bytes),
        }

        # Strip markdown fences if the model included them despite instructions
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.DOTALL).strip()

        try:
            data = json.loads(cleaned)
            detected_elements = [
                VisualObservation(**el)
                if isinstance(el, dict)
                else VisualObservation(description=str(el), category="general")
                for el in data.get("detected_elements", [])
            ]
            return MultimodalAnalysis(
                answer=str(data.get("answer", raw_text)),
                observations=[str(o) for o in data.get("observations", [])],
                detected_elements=detected_elements,
                uncertainty_notes=[str(u) for u in data.get("uncertainty_notes", [])],
                warnings=warnings,
                source_metadata=source_metadata,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Fallback: model did not return valid JSON — wrap raw text as answer
            warnings.append(
                "Gemini response was not in the expected JSON format; "
                "raw text returned as the answer."
            )
            return MultimodalAnalysis(
                answer=raw_text or "No response was generated.",
                observations=[],
                detected_elements=[],
                uncertainty_notes=[],
                warnings=warnings,
                source_metadata=source_metadata,
            )
