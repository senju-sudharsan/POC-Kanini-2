"""Persistent vector-store boundary; Chroma is an implementation detail."""

from typing import Protocol

from poc_kanini.rag.models import DocumentChunk, RetrievedChunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None: ...
    def delete_document(self, document_id: str) -> None: ...
    def similarity_search(self, embedding: list[float], limit: int, document_id: str | None = None) -> list[RetrievedChunk]: ...
    def all_chunks(self, document_id: str | None = None) -> list[DocumentChunk]: ...


class ChromaVectorStore:
    """Local, persistent Chroma collection for document chunks."""

    def __init__(self, persist_directory: str, collection_name: str = "document_chunks") -> None:
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError("ChromaDB is not installed. Install the backend dependencies.") from error
        client = chromadb.PersistentClient(path=persist_directory)
        self._collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding.")
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def similarity_search(self, embedding: list[float], limit: int, document_id: str | None = None) -> list[RetrievedChunk]:
        where = {"document_id": document_id} if document_id else None
        result = self._collection.query(query_embeddings=[embedding], n_results=limit, where=where, include=["documents", "metadatas", "distances"])
        ids, documents, metadata, distances = (result.get(key, [[]])[0] for key in ("ids", "documents", "metadatas", "distances"))
        return [RetrievedChunk(chunk=_chunk(chunk_id, text, values), distance=float(distance), score=1 - float(distance)) for chunk_id, text, values, distance in zip(ids, documents, metadata, distances)]

    def all_chunks(self, document_id: str | None = None) -> list[DocumentChunk]:
        result = self._collection.get(where={"document_id": document_id} if document_id else None, include=["documents", "metadatas"])
        return [_chunk(chunk_id, text, metadata) for chunk_id, text, metadata in zip(result["ids"], result["documents"], result["metadatas"])]


def _chunk(chunk_id: str, text: str, metadata: dict[str, str]) -> DocumentChunk:
    return DocumentChunk(chunk_id=chunk_id, document_id=metadata["document_id"], filename=metadata["filename"], document_type=metadata["document_type"], page_number=int(metadata["page_number"]), chunk_index=int(metadata["chunk_index"]), text=text, metadata=metadata)
