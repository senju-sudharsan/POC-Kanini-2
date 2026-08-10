# POC Kanini 2 — Project Guidance

## Product and scope

Build an Enterprise AI Assistant / Decision Agent using Data Science, Generative
AI, and Agentic AI. It is not a data warehouse and must not be redesigned into
a PDF-only chatbot. The document chatbot is a major capability within the
larger assistant.

The active implementation sequence and priorities are defined in
`docs/PROJECT_PLAN.md`; the current state is in `docs/CHECKPOINT.md`.

## Preservation constraints

- Do not modify `gemini-base/`; it is the original reference repository.
- Preserve the existing React/Vite UI, FastAPI shell, Gemini configuration, and
  LangGraph streaming/state patterns unless a scoped change requires adapting them.
- Do not delete or expand legacy Data Engineering artifacts automatically.
  Identify them for later archival/removal instead.

## Obsolete direction

Do not plan or implement enterprise warehouse architecture, raw or staging
layers, ETL pipelines, SCD Type 1/2, star schemas, OLAP, dimensional modelling,
or warehouse-oriented PostgreSQL architecture.

## Delivery principles

- Build incrementally: keep the core working before adding P1/P2 demonstrations.
- Use Gemini as the primary LLM and LangGraph as the core orchestration layer.
- Treat evidence/citations, safety, human approval, and state/memory as
  first-class design concerns when their phases begin.
- Keep modality-specific dependencies optional unless they are needed by the
  active feature.
