# Checkpoint - Phase 7 Complete

**Current phase:** Phase 7 — Memory, Checkpointing, Reflection, and Human-In-The-Loop.
**Status:** COMPLETE / VERIFIED (26/26 tests passed).
**Next feature:** Phase 8 — Insight, action workflows, and full frontend integration.

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
- Phase 7: thread-scoped MemorySaver checkpointing, short-term conversation memory,
  ReflectionDecision node, bounded error recovery, HITL approval boundary with
  approval_required/approval_id/approval_status state fields.

## Phase 7 delivery

**Module layout:** Phase 7 extends `backend/src/poc_kanini/graphs/` and
`backend/src/poc_kanini/models/`.

### New modules

| File | Role |
|------|------|
| `graphs/reflection.py` | `reflection_node(state)`: evaluates tool quality, detects errors, triggers bounded retry, enforces HITL approval boundary for controlled operations. Returns `ReflectionDecision` dict. |

### Updated modules

| File | Change |
|------|--------|
| `models/orchestration.py` | Added `ReflectionDecision(BaseModel)`, `ApprovalRequest(BaseModel)`, and Phase 7 state fields: `retry_count`, `max_retries`, `reflection`, `approval_required`, `approval_id`, `approval_reason`, `approval_status`. |
| `models/chat.py` | Added `thread_id` and `approval` fields to `ChatRequest`. Added `thread_id`, `approval_required`, `approval_id`, `approval_reason`, and `activities` to `ChatResponse`. |
| `graphs/chat.py` | All 5 specialists now route through `reflection_node` before synthesis. `MemorySaver` checkpointer compiled into graph. `route_reflection_decision()` router handles retry, HITL pause, cross-specialist, and synthesis routing. |
| `graphs/specialists.py` | `synthesize_node` now: returns `approval_required: True` in interrupted branch, `approval_required: False` in normal and approved branches, and the fallback answer includes prior conversation context when Gemini is unavailable. |
| `main.py` | `POST /api/chat` resolves `thread_id` (generate if missing), passes `config={"configurable": {"thread_id": thread_id}}` to `hybrid_chat_graph.ainvoke`, reads `approval_status` from request, and returns `thread_id` + `approval_*` fields in `ChatResponse`. |
| `frontend/src/App.tsx` | Tracks `threadId` state, passes it on subsequent requests, renders HITL approval banner with Approve/Reject buttons. |
| `frontend/src/lib/chat.ts` | `sendChat()` accepts `threadId`, `approval`, returns `thread_id`, `approval_required`, `approval_id`, `approval_reason`, `activities`. |

### Phase 7 state fields summary

| Field | Type | Purpose |
|-------|------|---------|
| `thread_id` | `str` | Thread/session identifier for checkpoint key |
| `retry_count` | `int` | Bounded retry counter per request |
| `max_retries` | `int` | Retry budget (default 1) |
| `reflection` | `dict` | `ReflectionDecision` output (quality_ok, needs_retry, reason) |
| `approval_required` | `bool` | True when graph paused for HITL |
| `approval_id` | `str \| None` | Unique ID for HITL approval request |
| `approval_reason` | `str \| None` | Human-readable explanation |
| `approval_status` | `str \| None` | "pending", "approved", or "rejected" |

### API changes

```
POST /api/chat
Request:
  {
    "messages": [...],
    "attachments": [...],          # unchanged
    "thread_id": "thread_xxx",     # NEW: optional, auto-generated if missing
    "approval": "approved"          # NEW: optional, for HITL resume
  }

Response:
  {
    "message": {"role": "assistant", "content": "..."},
    "thread_id": "thread_xxx",     # NEW: always returned
    "approval_required": false,    # NEW: true if HITL pause
    "approval_id": null,           # NEW: set when approval_required=true
    "approval_reason": null,       # NEW: set when approval_required=true
    "activities": [...]            # NEW: specialist activity log
  }
```

### Memory boundaries

| Type | Implementation | Phase |
|------|---------------|-------|
| **Short-term (in-scope)** | LangGraph `MemorySaver` per thread_id | Phase 7 |
| Long-term semantic memory | Vector-stored cross-session memories | Future |
| User profile memory | Per-user personalization store | Future |

### HITL Approval boundary

The HITL mechanism is a **mechanism demonstration**, not a live action system.

Controlled operation detection keywords: `sensitive`, `delete`, `production model`,
`requires approval`, `controlled operation`, `approve operation`.

Operations that reach HITL are bounded to the registered Phase 5 tools only.
Shell execution, arbitrary filesystem access, and arbitrary Python are never permitted.

## Validation completed

- `pytest backend/tests/test_phase7.py --basetemp=.pytest-tmp/tmp -v`: **26 passed**
- `pytest backend/tests --basetemp=.pytest-tmp/tmp -q`: full suite (Phases 1–7) — see regression results.
- `npm.cmd run build`: **passed** (1813 modules, ✓ built in 8.41s).

## Known limitations

- Gemini free-tier quota (20 req/day) means synthesis falls back to deterministic
  summaries during heavy testing. The fallback is production-safe.
- `MemorySaver` is in-process only — state is lost on server restart. SQLite-based
  persistence is the natural next step if required.
- HITL approval currently detects controlled operations via keyword matching.
  Phase 8 will introduce intent-based detection tied to actual action workflows.

## Exact next feature

Phase 8 — **Insight, action workflows, and full frontend integration.**
