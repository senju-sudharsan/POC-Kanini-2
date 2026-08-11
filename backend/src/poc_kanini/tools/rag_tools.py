"""Document evidence retrieval tool wrapping RagService.retrieve_evidence."""

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from poc_kanini.core.config import get_settings
from poc_kanini.rag.service import RagService
from poc_kanini.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class DocumentSearchInput(BaseModel):
    """Input schema for searching document evidence."""

    question: str = Field(
        ...,
        description="The natural language question to search for evidence in indexed documents.",
    )
    document_id: str | None = Field(
        default=None,
        description="Optional document ID to restrict search to a specific indexed document.",
    )


@tool("search_document_evidence", args_schema=DocumentSearchInput)
async def search_document_evidence(
    question: str,
    document_id: str | None = None,
) -> dict[str, Any]:
    """Retrieve relevant document evidence, chunk text, page numbers, and citations from indexed PDFs.

    Use this tool to search enterprise PDF documents for factual evidence snippets and source citations.
    The returned evidence is provided for agent reasoning and answer synthesis without triggering redundant LLM calls.
    """
    settings = get_settings()
    store = ChromaVectorStore(settings.rag_vector_store_dir)
    service = RagService(settings, store)

    try:
        sources, citations = await service.retrieve_evidence(question=question, document_id=document_id)
        return {
            "evidence": [
                {
                    "text": s.chunk.text,
                    "document_id": s.chunk.document_id,
                    "filename": s.chunk.filename,
                    "page_number": s.chunk.page_number,
                    "chunk_id": s.chunk.chunk_id,
                    "score": s.score,
                    "distance": s.distance,
                }
                for s in sources
            ],
            "citations": [c.model_dump() for c in citations],
            "retrieved_count": len(sources),
            "summary": (
                f"Retrieved {len(sources)} evidence snippet(s) from indexed documents."
                if sources
                else "No matching evidence found."
            ),
        }
    except Exception as error:
        logger.error("Error in search_document_evidence tool: %s", error)
        return {
            "error": str(error),
            "evidence": [],
            "citations": [],
            "retrieved_count": 0,
            "summary": "An error occurred while retrieving document evidence.",
        }
