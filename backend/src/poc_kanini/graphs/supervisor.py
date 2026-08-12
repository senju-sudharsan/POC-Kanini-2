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
Analyze the user request and any available image attachments, then select the single best specialist route:

- 'multimodal': The request includes an image attachment, or explicitly asks for visual analysis of an image.
- 'rag': The user asks for information, policies, or evidence contained in enterprise PDF documents.
- 'data': The user asks to profile, inspect, clean, or summarize a tabular dataset or records.
- 'ml': The user asks to train a machine learning model, evaluate ML metrics, or generate model predictions.
- 'general': General greeting, casual conversation, or queries that do not require tools.

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
        # 1. Immediate rule check for image attachments
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

        # 3. Fast keyword fallback check before LLM call if API key is unconfigured
        if not self._settings.gemini_api_key:
            return self._heuristic_route(user_query)

        # 4. LLM structured classification via Gemini
        try:
            client = genai.Client(api_key=self._settings.gemini_api_key)
            response = await client.aio.models.generate_content(
                model=self._settings.gemini_model,
                contents=f"User request: {user_query}",
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
        except Exception as error:
            logger.warning("Supervisor LLM routing failed, falling back to heuristic: %s", error)
            return self._heuristic_route(user_query)

    def _heuristic_route(self, query: str) -> RouteDecision:
        """Deterministic keyword classification fallback."""
        q = query.lower()
        if any(w in q for w in ["pdf", "document", "policy", "handbook", "evidence", "citation"]):
            return RouteDecision(route="rag", reason="Keyword match for document RAG", confidence=0.8)
        # 'profile' is checked before generic ML keywords so a combined
        # 'profile ... then train' query starts at the data specialist.
        if any(w in q for w in ["profile", "dataset", "columns", "rows", "csv", "tabular"]):
            return RouteDecision(route="data", reason="Keyword match for Data profiling", confidence=0.8)
        if any(w in q for w in ["train", "predict", "classifier", "regressor", "accuracy", "model"]):
            return RouteDecision(route="ml", reason="Keyword match for Machine Learning", confidence=0.8)
        if any(w in q for w in ["image", "photo", "picture", "visual", "chart"]):
            return RouteDecision(route="multimodal", reason="Keyword match for Multimodal", confidence=0.8)
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
