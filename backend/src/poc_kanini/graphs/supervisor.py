"""Structured LLM Supervisor router for the Phase 6 Hybrid Agent graph."""

import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types

from poc_kanini.core.config import Settings, get_settings
from poc_kanini.models.orchestration import ActivityEvent, AgentConversationState, RouteDecision

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_INSTRUCTION = """You are the Supervisor Router for AURA (Agentic Understanding & Retrieval Assistant).
Analyze the user request, any available image attachments, and any active attached documents, then select the single best specialist route:

- 'multimodal': The request includes an image attachment, or explicitly asks for visual analysis of an image.
- 'rag': The user asks for information, policies, or evidence contained in enterprise PDF documents, OR there are active attached documents and the request refers to them (e.g. "what does this document say", "what does this guy specialize at" when a document is attached).
- 'data': The user asks to profile, inspect, clean, or summarize a tabular dataset or records.
- 'ml': The user asks to train a machine learning model, evaluate ML metrics, or generate model predictions.
- 'general': General greeting, casual conversation, or queries that do not require tools and no documents are attached.

Output format (respond ONLY with valid JSON, no markdown fences):
{
  "route": "<rag|data|ml|multimodal|general>",
  "reason": "<brief explanation>",
  "confidence": 0.95
}"""


class SupervisorRouter:
    """Classifies incoming conversation requests to route to specialist nodes."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def route(self, state: AgentConversationState) -> RouteDecision:
        """Determine the route based on attachments and message history."""
        # 1. Immediate check for image attachments
        attachments = state.get("attachments") or []
        if attachments:
            return RouteDecision(
                route="multimodal",
                reason="Image attachment present in request.",
                confidence=1.0,
            )

        # 2. Extract latest user query text
        messages = state.get("messages") or []
        user_query = ""
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if content and not getattr(msg, "type", "").startswith("ai"):
                user_query = str(content)
                break

        if not user_query:
            return RouteDecision(
                route="general",
                reason="No user query provided; defaulting to general conversation.",
                confidence=1.0,
            )

        has_docs = bool(state.get("document_ids"))

        # 3. Route only high-confidence intents locally.  This keeps routine
        # requests from consuming a generative request while leaving Gemini as
        # the decision maker for genuinely ambiguous conversation.
        deterministic_decision = self._deterministic_route(user_query, has_documents=has_docs)
        if deterministic_decision:
            return deterministic_decision

        # 4. Fast keyword fallback check before LLM call if API key is unconfigured
        if not self._settings.gemini_api_key:
            return self._heuristic_route(user_query, has_documents=has_docs)

        # 5. LLM structured classification via Gemini for ambiguous requests.
        try:
            client = genai.Client(api_key=self._settings.gemini_api_key)
            doc_context = ""
            if state.get("document_ids"):
                doc_context = f"\n(Active attached document IDs: {', '.join(state['document_ids'])})"
            
            response = await client.aio.models.generate_content(
                model=self._settings.gemini_model,
                contents=f"User request: {user_query}{doc_context}",
                config=types.GenerateContentConfig(
                    system_instruction=SUPERVISOR_SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            raw_text = (response.text or "").strip()
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.DOTALL).strip()
            data = json.loads(cleaned)
            return RouteDecision(
                route=data.get("route", "general"),
                reason=data.get("reason", "Structured LLM classification"),
                confidence=float(data.get("confidence", 0.9)),
            )
        except Exception:
            logger.warning("Supervisor LLM routing unavailable; falling back to heuristic routing.")
            return self._heuristic_route(user_query, has_documents=has_docs)

    def _deterministic_route(self, query: str, has_documents: bool = False) -> RouteDecision | None:
        """Return a route for unambiguous intents, otherwise defer to Gemini.

        This deliberately uses a smaller set of cues than ``_heuristic_route``:
        an attached document alone is not enough to make an unrelated question
        a RAG question.
        """
        q = query.lower().strip()
        # Defer to Gemini/heuristic if the request combines multiple specialists (e.g., Data + ML)
        has_data = any(w in q for w in ("profile", "dataset", "columns", "csv", "tabular"))
        has_ml = any(w in q for w in ("train", "model", "classifier", "regressor", "accuracy", "predict"))
        if has_data and has_ml:
            return None

        if any(q.startswith(greeting) for greeting in (
            "hello", "hi", "hey", "how are you", "good morning", "good afternoon", "good evening",
        )):
            return RouteDecision(route="general", reason="Deterministic conversational greeting", confidence=1.0)
        if any(phrase in q for phrase in ("what can you do", "what do you do", "your capabilities")):
            return RouteDecision(route="general", reason="Deterministic capability request", confidence=1.0)
        if any(phrase in q for phrase in (
            "profile this dataset", "profile the dataset", "what columns", "show columns", "inspect this csv",
        )):
            return RouteDecision(route="data", reason="Deterministic dataset request", confidence=1.0)
        if any(phrase in q for phrase in (
            "train a model", "train the model", "train a classifier", "train a regressor", "model accuracy",
        )) or re.search(r"\bpredict\b", q):
            return RouteDecision(route="ml", reason="Deterministic machine-learning request", confidence=1.0)
        if any(phrase in q for phrase in (
            "analyze this image", "analyse this image", "analyze the image", "analyse the image",
        )):
            return RouteDecision(route="multimodal", reason="Deterministic image-analysis request", confidence=1.0)
        if has_documents and (
            re.search(r"\b(?:this|that|the|attached|uploaded)\s+(?:document|pdf|file|person|individual|guy|record)\b", q)
            or re.search(r"\b(?:according to|in)\s+(?:the\s+)?(?:document|pdf|file)\b", q)
            or "what does it say" in q
        ):
            return RouteDecision(route="rag", reason="Deterministic attached-document reference", confidence=1.0)
        return None

    def _heuristic_route(self, query: str, has_documents: bool = False) -> RouteDecision:
        """Deterministic keyword classification fallback."""
        q = query.lower().strip()
        if any(q.startswith(w) for w in ["hello", "hi", "hey", "how are you", "good morning", "good evening", "good afternoon"]):
            return RouteDecision(route="general", reason="Conversational greeting", confidence=1.0)
        if any(w in q for w in ["profile", "dataset", "columns", "rows", "csv", "tabular"]):
            return RouteDecision(route="data", reason="Keyword match for Data profiling", confidence=0.8)
        if any(w in q for w in ["train", "predict", "classifier", "regressor", "accuracy", "model"]):
            return RouteDecision(route="ml", reason="Keyword match for Machine Learning", confidence=0.8)
        if any(w in q for w in ["image", "photo", "picture", "visual", "chart"]):
            return RouteDecision(route="multimodal", reason="Keyword match for Multimodal", confidence=0.8)
        if has_documents or any(w in q for w in ["pdf", "document", "policy", "handbook", "evidence", "citation"]):
            return RouteDecision(route="rag", reason="Active document or keyword match for document RAG", confidence=0.8)
        return RouteDecision(route="general", reason="General conversation request", confidence=1.0)



async def supervisor_node(state: AgentConversationState) -> dict[str, Any]:
    """LangGraph node executing supervisor routing."""
    router = SupervisorRouter()
    decision = await router.route(state)
    activity = ActivityEvent(
        title="Supervisor Routing",
        data=f"Routed to '{decision.route}' specialist. Reason: {decision.reason}",
    )
    activities = list(state.get("activities") or [])
    activities.append(activity)

    return {
        "route": decision.route,
        "reason": decision.reason,
        "activities": activities,
        "step_count": state.get("step_count", 0) + 1,
    }
