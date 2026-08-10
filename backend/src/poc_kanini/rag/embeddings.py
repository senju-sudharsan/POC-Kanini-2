"""Gemini embeddings with explicit retrieval task types."""

from collections.abc import Sequence

from google import genai
from google.genai import types

from poc_kanini.core.config import Settings
from poc_kanini.rag.models import DocumentChunk


class GeminiEmbeddingService:
    """Embed chunks and queries using Gemini's retrieval-optimised modes."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def embed_documents(self, chunks: Sequence[DocumentChunk]) -> list[list[float]]:
        return await self._embed([chunk.text for chunk in chunks], "RETRIEVAL_DOCUMENT")

    async def embed_query(self, query: str) -> list[float]:
        values = await self._embed([query], "RETRIEVAL_QUERY")
        return values[0]

    async def _embed(self, contents: Sequence[str], task_type: str) -> list[list[float]]:
        if not contents:
            return []
        if not self._settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        client = genai.Client(api_key=self._settings.gemini_api_key)
        response = await client.aio.models.embed_content(
            model=self._settings.gemini_embedding_model,
            contents=list(contents),
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [list(item.values) for item in response.embeddings]
