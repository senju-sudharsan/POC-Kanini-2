# POC-Kanini-2 / AURA

**AURA (Agentic Understanding & Retrieval Assistant)** is a multimodal agentic AI assistant that understands documents, images, text, and structured datasets; retrieves grounded information; performs data analysis and machine learning; orchestrates specialised capabilities; maintains conversational context; and produces structured decision-support outputs.

POC-Kanini-2 is the engineering repository for AURA. It combines Data Science, Generative AI, and Agentic AI capabilities behind a React frontend, FastAPI backend, Gemini, and LangGraph. The document chatbot is a major feature, but not the whole product.

The initial migration preserves selected UI, FastAPI, Gemini configuration, and
LangGraph state patterns from `gemini-base/`. The reference directory is
read-only and must not be modified.

## Current implementation baseline

- `frontend/` — reusable React chat, activity timeline, Markdown, and evidence rendering.
- `backend/` — FastAPI shell, health endpoint, Gemini settings, and LangGraph contracts. All active Python implementation lives under [backend/src/poc_kanini/](file:///d:/POC-kanini-2/backend/src/poc_kanini/).
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
Q&A, and source citations. Phase 4A adds a supervised Machine Learning engine:
dataset profiling, sklearn-based preprocessing pipelines, classification and
regression training, evaluation metrics, feature importance, and prediction
through a process-lifetime model cache. Phase 4B adds Gemini multimodal
understanding: image validation (JPEG/PNG/WEBP, 10 MB limit), Gemini
multimodal content generation, structured visual analysis with typed observations,
category-tagged detected elements, explicit uncertainty notes, and graceful
JSON-fallback handling. Phase 5 adds an independent Tool Layer (`search_document_evidence`,
`profile_dataset_tool`, `train_ml_model_tool`, `predict_ml_model_tool`, `analyze_image_tool`)
and tool registry (`ALL_TOOLS`), creating clean tool interfaces with explicit input/output
contracts and strict security boundaries for future Phase 6 LangGraph agent routing.

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

`POST /api/ml/profile` accepts a list of JSON row records and returns a
`DatasetProfile` (counts, types, missing values, summary statistics).
`POST /api/ml/train` trains a supervised classifier or regressor on the supplied
dataset and returns a `TrainResponse` containing a process-lifetime `model_id`,
metrics, feature importance, and preprocessing summary. `POST /api/ml/predict`
runs predictions using a cached trained model; models are held in a
process-lifetime in-memory cache keyed by UUID and disappear on server restart.
`POST /api/multimodal/analyze` accepts a multipart image upload (JPEG, PNG, or
WEBP, up to 10 MB) with an optional `question` field; sends the image to Gemini
using `types.Part.from_bytes()` and returns a structured `MultimodalAnalysis`
containing an answer, visual observations, category-tagged detected elements,
uncertainty notes, and source metadata. Models return uncertainty as text notes
rather than fabricated confidence scores.

Phase 5 adds an independent Tool Layer (`search_document_evidence`,
`profile_dataset_tool`, `train_ml_model_tool`, `predict_ml_model_tool`, `analyze_image_tool`)
and tool registry (`ALL_TOOLS`). Phase 6 adds a flat hybrid LangGraph StateGraph with
supervisor routing, 5 specialist nodes, bounded cross-specialist transitions, and
citation-preserving synthesis. Phase 7 adds thread-scoped `MemorySaver` checkpointing,
short-term memory context, reflection node error recovery, and human-in-the-loop (HITL)
approval boundaries.

Phase 8 completes the unified enterprise assistant product experience:
- **Unified Assistant Interface (`POST /api/chat`)**: A single conversational entry point routing text, documents, images, datasets, ML training, predictions, and reports.
- **Document Upload & Thread Association**: PDF uploads indexed via `/api/documents/index` are automatically associated with current thread context for grounded answer generation with `[filename — Page X]` citations.
- **Multimodal Chat**: Attach JPEG, PNG, or WEBP images directly in chat as Base64 payloads; analyzed by `analyze_image_tool` with observations rendered in message timeline.
- **Structured Tool Result Cards**: Frontend renders dedicated cards for evidence citations, ML evaluation metrics grids (Accuracy, F1, Precision, Recall, MAE, R²), visual observations, and warnings.
- **Structured Domain Reports (`POST /api/reports/generate`)**: Synthesizes structured `ReportPayload` cards (executive summaries, dataset analysis, document intelligence, visual inspection reports).
- **Controlled Action Abstraction (`POST /api/actions/execute`)**: Safe local action abstraction (`ActionRequest`/`ActionResult`) enforcing human approval boundaries when controlled operations are detected.

ChromaDB stores local vectors in `data/chroma` by default. Set
`GEMINI_EMBEDDING_MODEL`, `RAG_VECTOR_STORE_DIR`, and `RAG_TOP_K` only when
overriding their defaults.

## Phase 9 — Production Hardening (complete)

Phase 9 hardened the system for deployment readiness without redesigning the frontend or replacing any working architecture.

**Changes in Phase 9:**

- **CORS middleware** added with restrictive allowlist (`cors_origins` setting, overrideable via env).
- **Typed report request model** — `POST /api/reports/generate` now validates the request body through Pydantic, rejecting unknown `report_type` values with 422.
- **Provider error classification** — Gemini errors are mapped to correct HTTP codes: 429 rate limit, 401 auth failure, 504 timeout, 503 unavailable. Raw stack traces never reach the client.
- **Structured startup logging** — root logger configured at `INFO`, noisy third-party loggers suppressed.
- **Filename sanitisation** — all upload endpoints strip directory components from client-supplied filenames, preventing path traversal.
- **35 new E2E integration tests** covering all 9 verification flows (chat schema, RAG citations, multimodal validation, data profiling, ML round-trip, memory isolation, HITL, reports, and actions).
- **Docker hardened** — multi-stage build, `.dockerignore`, `docker-compose.yml` with named Chroma volume and health check.

## Docker quickstart

```bash
# Copy environment template and add your Gemini API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=...

# Build and start
docker compose up --build

# AURA is served at http://localhost:8000/app
# API at http://localhost:8000/api/
```

The Chroma vector store is persisted in a named Docker volume (`aura_chroma`) so indexed documents survive container restarts.

## Test results (Phase 9)

| Suite | Tests | Result |
|---|---|---|
| Phases 1–8 (regression) | 123 | ✓ passed |
| Phase 9 (new E2E + security) | 35 | ✓ passed |
| **Total** | **158** | **✓ 0 failures** |
| Frontend build | — | ✓ passed |

## Security boundaries

- Uploaded files: type-validated, size-limited (20 MiB), filename sanitised — no path traversal
- Base64 image attachments only — no external URL fetching
- Controlled actions only: no shell execution, no subprocess, no arbitrary filesystem access, no external HTTP, no database mutation
- CORS: explicit allowlist — not wildcard
- Error responses: provider errors mapped to HTTP status — no raw stack traces or API keys exposed
