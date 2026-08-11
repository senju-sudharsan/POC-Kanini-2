# Checkpoint - Phase 6 Complete

**Current phase:** Phase 6 — Hybrid LangGraph Agent Architecture.
**Status:** COMPLETE / VERIFIED.
**Next feature:** Phase 7 — Human-in-the-loop approval, safety guardrails, and streaming SSE responses.

## Completed foundations

- Phase 1: Gemini / LLM chat through FastAPI and a minimal LangGraph workflow.
- Phase 2: safe PDF ingestion, extraction, OCR fallback, structural parsing,
  classification, and provenance-preserving structured output.
- Phase 3: page-aware chunking, Gemini retrieval embeddings, local persistent
  ChromaDB abstraction, hybrid retrieval, grounded answers, and citations.
- Phase 4A: supervised ML engine with dataset profiling, preprocessing,
  classification, regression, evaluation, prediction, and feature importance.
- Phase 4B: image validation, Gemini multimodal understanding, structured
  visual analysis, and JSON parsing with graceful fallback.
- Phase 5: independent tool layer wrapping RAG, data profiling, ML training/prediction,
  and multimodal analysis with explicit schemas, security boundaries, and registry.
- Phase 6: flat hybrid LangGraph StateGraph with structured supervisor routing,
  five specialist nodes, bounded cross-specialist transitions, multimodal attachment
  propagation, and citation-preserving synthesis.

## Phase 6 delivery

**Module layout:** All Phase 6 code lives under `backend/src/poc_kanini/graphs/` and
`backend/src/poc_kanini/models/`.

### New modules

| File | Role |
|------|------|
| `graphs/supervisor.py` | `SupervisorRouter` + `supervisor_node`: classifies each request via Gemini structured output (`RouteDecision`). Falls back to deterministic keyword heuristic on API error or missing key. |
| `graphs/specialists.py` | Five specialist nodes (`support_agent_node`, `data_agent_node`, `ml_agent_node`, `multimodal_agent_node`, `general_agent_node`) plus `synthesize_node`. |
| `graphs/chat.py` | Assembles `StateGraph(AgentConversationState)` with conditional edges, cross-specialist routing, and `max_steps = 5` safety boundary. Exports `hybrid_chat_graph`. |
| `graphs/__init__.py` | Package re-export. |

### Updated modules

| File | Change |
|------|--------|
| `models/orchestration.py` | Added `RouteDecision(BaseModel)` and `AgentConversationState(TypedDict)`. |
| `models/chat.py` | Added `ImageAttachment(BaseModel)` and optional `attachments` field on `ChatRequest`. |
| `main.py` | `POST /api/chat` now invokes `hybrid_chat_graph` with `messages` + `attachments` + `step_count` + `max_steps`. Backwards compatible with text-only requests. |

### Specialist routing table

| Route | Specialist Node | Tools |
|-------|----------------|-------|
| `rag` | `support_agent_node` | `search_document_evidence` |
| `data` | `data_agent_node` | `profile_dataset_tool` → optional cross-specialist to ML |
| `ml` | `ml_agent_node` | `train_ml_model_tool`, `predict_ml_model_tool` |
| `multimodal` | `multimodal_agent_node` | `analyze_image_tool` |
| `general` | `general_agent_node` | (none — direct Gemini conversation) |

### Cross-specialist transitions

- `data → ml`: triggered when query contains training/classifier/regression keywords AND `step_count < max_steps`.
- Hard `max_steps = 5` bound prevents infinite loops.

### Synthesis

`synthesize_node` calls Gemini with all accumulated tool results as grounded context.
Falls back to a deterministic citation-preserving summary on any API error (e.g. quota exhaustion).
Citation format preserved: `[filename — Page X]`.

## Validation completed

- `pytest backend/tests/test_graphs.py --basetemp=.pytest-tmp/tmp -v`: **18 passed** (all Phase 6).
- `pytest backend/tests --basetemp=.pytest-tmp/tmp -q`: full suite (Phases 1–6) — see task log.
- Frontend build: unchanged (`npm.cmd run build` was validated in Phase 5).

## Exact next feature

Phase 7 — **Human-in-the-loop approval, safety guardrails, and optional streaming SSE responses.**
