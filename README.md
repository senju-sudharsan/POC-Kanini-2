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

Phase 5 introduces standard LangChain `@tool` specifications in `backend/src/poc_kanini/tools/`:
- `search_document_evidence`: RAG evidence retrieval and citations.
- `profile_dataset_tool`: Tabular dataset profiling.
- `train_ml_model_tool`: Supervised ML classification/regression model training.
- `predict_ml_model_tool`: Predictions via cached model UUIDs.
- `analyze_image_tool`: Multimodal image analysis using base64 image strings.
All tools enforce strict security boundaries (rejecting unvalidated local file paths) and are registered in `ALL_TOOLS` for future Phase 6 LangGraph agent tool binding.

ChromaDB stores local vectors in `data/chroma` by default. Set
`GEMINI_EMBEDDING_MODEL`, `RAG_VECTOR_STORE_DIR`, and `RAG_TOP_K` only when
overriding their defaults.
