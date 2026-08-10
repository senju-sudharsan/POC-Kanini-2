# POC Kanini 2

POC Kanini 2 is being developed as an Enterprise AI Assistant / Decision Agent.
It combines Data Science, Generative AI, and Agentic AI capabilities behind a
React frontend, FastAPI backend, Gemini, and LangGraph. The document chatbot is
a major feature, but not the whole product.

The initial migration preserves selected UI, FastAPI, Gemini configuration, and
LangGraph state patterns from `gemini-base/`. The reference directory is
read-only and must not be modified.

## Current implementation baseline

- `frontend/` — reusable React chat, activity timeline, Markdown, and evidence rendering.
- `backend/` — FastAPI shell, health endpoint, Gemini settings, and minimal shared LangGraph state contract.
- `gemini-base/` — original reference repository; do not modify.

Phase 1A is implemented: text-only enterprise chat is validated by FastAPI,
orchestrated through a compact LangGraph workflow, and answered by Gemini. The
UI retains conversation state, progress activities, cancellation, and Markdown
responses. Phase 1B supplies reusable, tested NLP foundations for file handling,
normalization, linguistic processing, and embeddings. Phase 2 adds safe PDF
processing: validation, page-level extraction/provenance, lightweight layout
signals, optional local OCR fallback, classification, and a basic upload/status
control. Phase 3 adds deterministic page-aware chunking, Gemini retrieval
embeddings, persistent local ChromaDB storage, hybrid retrieval, grounded PDF
Q&A, and source citations. ML, specialist agents, memory, and human approval
remain intentionally deferred.

## Active direction

The active architecture routes enterprise questions through LangGraph to
support/RAG, data-analysis, and ML specialist capabilities, producing
evidence-backed insights, recommendations, human-approved actions, and reports.
See [the project plan](docs/PROJECT_PLAN.md),
[architecture](docs/ARCHITECTURE.md), and [current checkpoint](docs/CHECKPOINT.md)
for the source of truth.

The prior enterprise data-warehouse direction (raw/staging layers, ETL, SCD,
star schema, OLAP, dimensional modelling, and warehouse PostgreSQL design) is
obsolete and must not be expanded. No such implementation was found outside the
reference repository during the scope reset.

## Local checks

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Create a virtual
environment with `python -m venv .venv`, activate it, then install the backend
with `.venv\\Scripts\\python -m pip install -e ./backend` on Windows.

Run the backend with `.venv\\Scripts\\uvicorn poc_kanini.main:app --app-dir backend/src --reload`.
Build the frontend with `npm install` then `npm run build` from `frontend/`.
During development Vite proxies `/api` to the FastAPI server; the production
Docker image serves the built UI at `/app`.

`POST /api/chat` accepts text conversation history and returns one Gemini answer.
`POST /api/documents/process` accepts one PDF upload and returns page-level
structured processing output. `POST /api/documents/index` processes and indexes
a PDF into local ChromaDB. `POST /api/documents/chat` answers an indexed
document question with retrieved-source metadata and `[filename - Page X]`
citations.

ChromaDB stores local vectors in `data/chroma` by default. Set
`GEMINI_EMBEDDING_MODEL`, `RAG_VECTOR_STORE_DIR`, and `RAG_TOP_K` only when
overriding their defaults.
