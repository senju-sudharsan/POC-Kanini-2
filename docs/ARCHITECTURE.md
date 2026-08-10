# Architecture - Enterprise AI Assistant / Decision Agent

## Target architecture

```text
User -> Enterprise AI Assistant -> LangGraph
  |- Support / RAG agent -> PDFs, documents, OCR, retrieval
  |- Data agent -> structured files and Python analysis tools
  `- ML agent -> trained ML models and predictions
```

The document chatbot is a major assistant capability, not the entire product.
LangGraph will progressively own routing, planning, state, and specialist
collaboration as those phases begin.

## Implemented document intelligence path

```text
PDF upload -> Phase 2 processing -> page/structure-aware chunking
-> Gemini RETRIEVAL_DOCUMENT embeddings -> local ChromaDB
-> semantic + keyword retrieval -> Gemini grounded answer -> citations
```

Phase 2 preserves each page's number, original/normalized text, semantic terms,
extraction method, OCR state, and layout signals. Phase 3 consumes that contract
directly and creates deterministic `DocumentChunk` records without duplicating
document models.

`GeminiEmbeddingService` applies `RETRIEVAL_DOCUMENT` to chunks and
`RETRIEVAL_QUERY` to user questions. `ChromaVectorStore` is a persistent local
implementation behind a vector-store abstraction. It stores embeddings, chunk
text, IDs, and provenance metadata in `data/chroma` by default, supporting
upsert, document deletion, similarity search, and document filtering.

Retrieval merges semantic candidates with keyword scores and deduplicates by
chunk ID. Each result retains its chunk, distance, filename, page number, and
document ID. The RAG instruction requires retrieved evidence, acknowledgement
of insufficiency, real citations, and no chain-of-thought disclosure.

## APIs and UI

- `POST /api/chat` remains the Phase 1 text-only chat path.
- `POST /api/documents/process` remains the Phase 2 structured processing path.
- `POST /api/documents/index` processes and indexes one PDF.
- `POST /api/documents/chat` returns an answer, citations, and retrieved source
  metadata for an indexed document question.

The React upload control indexes a PDF; subsequent chat questions use the
document-aware route and show citations in the form `[filename - Page X]`.

## Non-goals

This system does not contain an enterprise data warehouse, raw/staging layers,
ETL pipelines, SCD processing, star schemas, OLAP, dimensional models, or a
warehouse-focused PostgreSQL design. Phase 4+ capabilities are not implemented.
