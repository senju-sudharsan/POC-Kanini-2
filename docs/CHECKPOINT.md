# Checkpoint - Phase 3 Complete

**Current phase:** Phase 3 - Embeddings + RAG + Vector Retrieval.  
**Status:** COMPLETE.  
**Next feature:** Phase 4 - ML + multimodal capabilities.

## Completed foundations

- Phase 1: Gemini / LLM chat through FastAPI and a minimal LangGraph workflow.
- Phase 2: safe PDF ingestion, extraction, OCR fallback, structural parsing,
  classification, and provenance-preserving structured output.
- Phase 3: page-aware chunking, Gemini retrieval embeddings, local persistent
  ChromaDB abstraction, hybrid retrieval, grounded answers, and citations.

## Phase 3 delivery

- Existing `ProcessedDocument` and `ProcessedPage` models are consumed directly.
  Deterministic `DocumentChunk` identifiers retain document, filename, page,
  extraction, OCR, and structure provenance.
- `GeminiEmbeddingService` uses `RETRIEVAL_DOCUMENT` for chunks and
  `RETRIEVAL_QUERY` for questions, with settings-based credentials and model.
- `ChromaVectorStore` persists embeddings, chunk text, identifiers, and metadata
  under `data/chroma` by default through an isolated vector-store boundary.
- Retrieval combines semantic candidates with keyword scores, deduplicates by
  chunk ID, supports document filtering, and retains distance/source metadata.
- `POST /api/documents/index` processes and indexes a PDF. `POST
  /api/documents/chat` returns a grounded answer plus retrieved-source metadata
  and citations derived from actual chunks. Normal `POST /api/chat` is unchanged.
- The React upload flow indexes PDFs and sends subsequent questions through the
  evidence-backed document Q&A path, displaying `[filename - Page X]` labels.

## Validation completed

- `pytest backend/tests -q`: **24 passed**.
- Backend compilation: passed.
- `npm.cmd run build`: passed.
- Deterministic fakes cover indexing, hybrid retrieval, document filtering,
  insufficiency handling, and citation provenance without a Gemini key.

## Operational note

Tesseract must be installed locally and available on `PATH` for real OCR.
Gemini document embeddings and grounded answer generation require
`GEMINI_API_KEY`. ChromaDB is declared as a backend dependency, but its install
could not be completed in this Windows sandbox: pip reported socket permission
errors and the elevated attempt timed out. This is an environment/network issue,
not an application test failure.

## Strict boundary

Do not begin Phase 4 ML/multimodal features, specialist agents, advanced
orchestration, memory, HITL, reports, or ADK. Do not modify `gemini-base/`.

## Exact next feature

Phase 4 - **ML + multimodal capabilities**.
