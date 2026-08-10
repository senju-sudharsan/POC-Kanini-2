"""RAG-ready document chunk models with citation provenance."""

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A retrieval-ready piece of a processed document."""

    chunk_id: str
    document_id: str
    filename: str
    document_type: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class Citation(BaseModel):
    """A user-visible reference derived from a retrieved document chunk."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(ge=1)
    label: str


class RetrievedChunk(BaseModel):
    """A chunk and its retrieval score, retaining all source provenance."""

    chunk: DocumentChunk
    distance: float | None = None
    keyword_score: float = 0.0
    score: float = 0.0


class RagResponse(BaseModel):
    """Grounded document answer with its actual retrieved evidence."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_sources: list[RetrievedChunk] = Field(default_factory=list)
