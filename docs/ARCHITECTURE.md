# Architecture - Enterprise AI Assistant / Decision Agent

## Target architecture

```text
User → POST /api/chat (text + optional image attachment)
  → Phase 6 Hybrid LangGraph StateGraph
      ├── supervisor_node (RouteDecision via Gemini structured output or heuristic)
      │     routes to one of:
      ├── support_agent_node   → search_document_evidence → RAG evidence
      ├── data_agent_node      → profile_dataset_tool      → optional cross-specialist → ml
      ├── ml_agent_node        → train/predict_ml_model_tool
      ├── multimodal_agent_node→ analyze_image_tool        → visual observations
      └── general_agent_node   → (no tools, direct Gemini chat)
      └── synthesize_node      → Gemini grounded final answer with citations
```

The document chatbot is a major assistant capability, not the entire product.
LangGraph owns routing, bounded execution (max_steps=5), specialist collaboration,
and final synthesis with citation preservation.

## Active implementation layout

All active backend logic lives within the consolidated python package under [`backend/src/poc_kanini/`](file:///d:/POC-kanini-2/backend/src/poc_kanini/):
- `documents/` — PDF extraction, parsing, OCR fallback, and structured output.
- `nlp/` — Text cleaning, normalization, and tokenization.
- `rag/` — Chunking, embeddings, vector database, and hybrid retrieval.
- `ml/` — Dataset profiling, preprocessing, supervised model training, evaluation, prediction, and feature importance.
- `multimodal/` — Image validation, Gemini multimodal content generation, structured visual analysis.
- `tools/` — Independently testable tools (RAG evidence, dataset profiling, ML training/prediction, multimodal analysis) & tool registry.
- `graphs/` — LangGraph agent routing and state workflows.
- `services/` — Clients and wrappers for LLM APIs (Gemini).
- `models/` — Pydantic structures and schemas.

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

## Implemented ML / Data Science path (Phase 4A)

```text
Tabular dataset (JSON records)
  -> profile_dataframe() -> DatasetProfile
  -> MlPreprocessor (median imputation + scaling for numeric;
                     most-frequent imputation + one-hot for categorical)
  -> train/test split
  -> LogisticRegression / RandomForestClassifier (classification)
     LinearRegression   / RandomForestRegressor  (regression)
  -> evaluation metrics (accuracy/F1 or MAE/RMSE/R²)
  -> feature importance (coefficients or feature_importances_)
  -> TrainResponse (model_id, metrics, features, warnings)
  -> MlService cache (process-lifetime, UUID-keyed)
  -> prediction via /api/ml/predict
```

Models disappear when the FastAPI process restarts. The `_save_pipeline` /
`_load_pipeline` abstraction in `MlService` is the extension point for future
persistent or cloud-backed model storage.

## Implemented Multimodal path (Phase 4B)

```text
Image upload (JPEG / PNG / WEBP, ≤10 MB)
  → validate_image() → ImageValidationError on bad MIME / empty / too large
  → Part.from_bytes(data=bytes, mime_type=mime_type)  [google-genai SDK]
  → Gemini generate_content (image part + user question part)
     system instruction: observe → infer → cite evidence → flag uncertainty
  → JSON response parsed into MultimodalAnalysis
     ├─ answer
     ├─ observations []
     ├─ detected_elements [] (description + category)
     ├─ uncertainty_notes []
     └─ source_metadata {filename, mime_type, size_bytes}
  → Graceful fallback to raw text if JSON cannot be parsed
```

`GeminiMultimodalService` reuses the existing `genai.Client` pattern.
No additional model or Gemini client was introduced. The current
`gemini-2.5-flash` model supports multimodal (image + text) natively.
Uncertainty is represented as explicit `uncertainty_notes` strings, not
invented numerical confidence scores.

## Implemented Tool Layer (Phase 5)

```text
tools/
  ├── rag_tools.py          → search_document_evidence (wraps RagService.retrieve_evidence)
  ├── data_tools.py         → profile_dataset_tool (wraps MlService.profile)
  ├── ml_tools.py           → train_ml_model_tool & predict_ml_model_tool (wraps MlService)
  ├── multimodal_tools.py   → analyze_image_tool (wraps MultimodalService)
  └── __init__.py           → ALL_TOOLS registry & TOOLS_BY_NAME mapping
```

All tools use LangChain `@tool` specifications with explicit Pydantic input schemas,
JSON-serializable return contracts, docstrings for LLM tool calling, and strict
security boundaries (no arbitrary local file path loading). They are ready for
binding to Phase 6 LangGraph agent routing.

## APIs and UI

- `POST /api/chat` — **Phase 6 Hybrid Agent**: accepts `ChatRequest` with `messages` and optional
  `attachments` (base64 image list). Routed through the full `hybrid_chat_graph` StateGraph with supervisor
  routing, specialist tool execution, and Gemini-grounded synthesis. Fully backwards compatible with
  text-only requests.
- `POST /api/documents/process` remains the Phase 2 structured processing path.
- `POST /api/documents/index` processes and indexes one PDF.
- `POST /api/documents/chat` returns an answer, citations, and retrieved source
  metadata for an indexed document question.
- `POST /api/ml/profile` accepts a list of JSON records and returns a `DatasetProfile`.
- `POST /api/ml/train` trains a classifier or regressor and returns a `TrainResponse`
  containing a process-lifetime `model_id`, evaluation metrics, and feature importance.
- `POST /api/ml/predict` runs predictions using a cached trained model.
- `POST /api/multimodal/analyze` accepts a multipart image file (JPEG/PNG/WEBP, ≤10 MB)
  and an optional question string; returns a structured `MultimodalAnalysis` with
  Gemini's visual observations, detected elements, uncertainty notes, and source metadata.

The React upload control indexes a PDF; subsequent chat questions use the
document-aware route and show citations in the form `[filename - Page X]`.

## Non-goals

This system does not contain an enterprise data warehouse, raw/staging layers,
ETL pipelines, SCD processing, star schemas, OLAP, dimensional models, or a
warehouse-focused PostgreSQL design.

Phase 7+ features (HITL approval nodes, safety guardrails, streaming SSE,
persistent memory, and multi-turn autonomous planning) are not yet implemented.

## Implemented Agent Graph (Phase 6)

```text
graphs/
  ├── supervisor.py    → SupervisorRouter (Gemini RouteDecision + keyword heuristic fallback)
  │                      supervisor_node(state) → {route, reason, step_count, activities}
  ├── specialists.py   → support_agent_node   (search_document_evidence)
  │                      data_agent_node      (profile_dataset_tool + cross-specialist to ml)
  │                      ml_agent_node        (train_ml_model_tool / predict_ml_model_tool)
  │                      multimodal_agent_node(analyze_image_tool)
  │                      general_agent_node   (direct Gemini conversation)
  │                      synthesize_node      (Gemini grounded answer + citation fallback)
  └── chat.py          → hybrid_chat_graph: StateGraph(AgentConversationState)
                         START → supervisor → [specialist] → synthesize → END
                         cross-specialist: data → ml (bounded by max_steps = 5)

models/
  ├── orchestration.py → RouteDecision(BaseModel), AgentConversationState(TypedDict)
  └── chat.py          → ImageAttachment(BaseModel), ChatRequest.attachments (optional)
```

**Synthesis guarantees:**
- Gemini available: full grounded response using all accumulated tool outputs as context.
- Gemini 429/unavailable: deterministic fallback summary extracted from `tool_results`,
  including preserved citation labels `[filename — Page X]`.

**Security boundaries (Phase 6):**
- Attachments are passed as base64 strings; no filesystem paths accepted from callers.
- `max_steps = 5` hard limit prevents unbounded specialist loops.
- Each specialist wraps tool errors as structured payloads — no stack traces propagate.
