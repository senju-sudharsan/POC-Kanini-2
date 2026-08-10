"""Hybrid provenance-preserving document retrieval."""

import re

from poc_kanini.rag.embeddings import GeminiEmbeddingService
from poc_kanini.rag.models import DocumentChunk, RetrievedChunk
from poc_kanini.rag.vector_store import VectorStore


class RetrievalService:
    def __init__(self, embeddings: GeminiEmbeddingService, store: VectorStore) -> None:
        self._embeddings, self._store = embeddings, store

    async def retrieve(self, query: str, top_k: int = 5, document_id: str | None = None) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        semantic = self._store.similarity_search(await self._embeddings.embed_query(query), max(top_k * 3, top_k), document_id)
        keywords = set(re.findall(r"[a-z0-9]+", query.casefold()))
        candidates: dict[str, RetrievedChunk] = {result.chunk.chunk_id: result for result in semantic}
        for chunk in self._store.all_chunks(document_id):
            hits = sum(1 for token in keywords if token in chunk.text.casefold())
            if hits:
                current = candidates.get(chunk.chunk_id, RetrievedChunk(chunk=chunk))
                current.keyword_score = hits / max(len(keywords), 1)
                candidates[chunk.chunk_id] = current
        for result in candidates.values():
            semantic_score = max(result.score, 0.0)
            result.score = 0.75 * semantic_score + 0.25 * result.keyword_score
        return sorted(candidates.values(), key=lambda item: item.score, reverse=True)[:top_k]
