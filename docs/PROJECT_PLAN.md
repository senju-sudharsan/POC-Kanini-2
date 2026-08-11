# Project Plan — Enterprise AI Assistant / Decision Agent

## Product objective

Deliver an enterprise assistant that turns a user question into evidence,
reasoning, insight, recommendation or action, human approval when needed, and
a report. The assistant will route work through LangGraph to specialist
support/RAG, data-analysis, and ML capabilities.

The PDF/document chatbot is a required standout feature, not the whole product.
It will support document ingestion, parsing/OCR, chunking, embeddings, vector
retrieval, Gemini-grounded answers, and evidence/citations.

## Active curriculum alignment

1. Text, image, and audio processing foundations
2. Embeddings, representations, and GenAI basics
3. Advanced Google GenAI capabilities
4. Intelligent document processing systems
5. Vector databases and retrieval systems
6. Multimodal AI with Gemini
7. Foundations of autonomous agents
8. Agent function calling and tooling
9. Agent memory and state management
10. LangGraph state graph architectures
11. Google ADK and ecosystem

## Development phases

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Text/NLP + Gemini foundation | COMPLETE |
| 2 | Intelligent PDF/document processing | COMPLETE |
| 3 | Embeddings + RAG + vector retrieval | COMPLETE |
| 4 | ML + multimodal capabilities | COMPLETE |
| 5 | Tools + Support/RAG/Data/ML specialist capabilities | COMPLETE |
| 6 | LangGraph orchestration + planning + reasoning | COMPLETE |
| 7 | Memory + checkpointing + reflection + human-in-the-loop | COMPLETE |
| 8 | Insight + action + reports + full frontend integration | NEXT |
| 9 | Google ADK / Vertex / Google ecosystem demonstrations | PLANNED |
| 10 | Testing + evaluation + Docker + documentation + final demo | PLANNED |

## Priorities

| Priority | Required scope |
| --- | --- |
| P0 — must work | Gemini; NLP foundation; PDF ingestion; OCR; embeddings; RAG/vector retrieval; evidence/citations; an ML model; data analysis; specialist capabilities; LangGraph; tool/function calling; agent routing; memory; human-in-the-loop; React frontend; FastAPI backend |
| P1 — strongly desired | FAISS; ChromaDB; hybrid search; cross-encoder reranking; multimodal PDF understanding; web search; long-term semantic memory; reflection; error recovery; reports; checkpointing |
| P2 — stretch | Pinecone; audio; video; graph-database memory; context caching; model tuning; Vertex deployment; Google Cloud extensions; advanced ADK |

When time is constrained, protect P0 before P1 and P1 before P2.

## Technology direction

Gemini is the primary LLM. Demonstrations should progressively include system
instructions, prompting, structured JSON, function calling, safety settings,
and grounding. Use one working vector system first; FAISS or ChromaDB are
likely candidates. Additional vector technologies must have a meaningful
demonstration rather than duplicate production systems.

Google ADK and Vertex-related work comes after the core LangGraph application
is stable, so it cannot destabilize the working assistant.

## Retired scope

The former Data Engineering plan is obsolete: enterprise data warehouse, raw
and staging layers, ETL, SCD Types 1 and 2, star schema, OLAP, dimensional
modelling, and warehouse-oriented PostgreSQL architecture are out of active
scope. Obsolete directories and database schemas under root folders have been removed. Any remaining reference material under `gemini-base/` is preserved for reference only.
