"""Structured LLM Supervisor router for the Phase 6 Hybrid Agent graph."""

import json
import logging
import re
import uuid
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
- 'data': The user asks to visualize, chart, plot, graph, generate charts (bar, line, scatter, pie, donut, KPI metrics), profile, inspect, clean, or summarize a tabular dataset or records.
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

        # Check for explicit approval resumption
        if state.get("approval_status") == "approved" or user_query.lower().startswith("proceed") or "[decision: approved]" in user_query.lower():
            op = state.get("operation") or "ml"
            return RouteDecision(
                route=op if op in ("ml", "data", "rag", "multimodal") else "ml",
                reason="Resumed approved operation",
                confidence=1.0,
            )

        has_docs = bool(state.get("document_ids"))

        # 3. Route only high-confidence intents locally.
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

        Precedence:
        1. Explicit image analysis -> multimodal
        2. Explicit visualization / charting intent -> data
        3. Explicit controlled ML train / predict intent -> ml
        4. Explicit dataset profiling / CSV intent -> data
        5. Deterministic conversational / identity intents -> general
        6. Explicit document / section reference intent -> rag
        7. Return None to allow Gemini supervisor routing
        """
        q = query.lower().strip()

        # 1. Explicit visual image analysis
        if any(phrase in q for phrase in (
            "analyze this image", "analyse this image", "analyze the image", "analyse the image",
        )):
            return RouteDecision(route="multimodal", reason="Deterministic image-analysis request", confidence=1.0)

        # 2. Explicit dataset profiling intent (including cross-specialist profiling -> ML workflows)
        has_explicit_profiling = any(phrase in q for phrase in (
            "profile this", "profile the", "profile dataset", "profile tabular", "inspect this csv", "what columns", "show columns",
        )) or bool(re.search(r"\bprofile\s+(?:this|the|tabular|dataset|csv)\b", q))

        # 3. Explicit visualization / charting intent (takes precedence over generic dataset/ML heuristics when no explicit training/prediction is requested)
        has_explicit_viz = any(phrase in q for phrase in (
            "plot ", "chart", "visualize", "visualise", "visualization", "visualisation",
            "histogram", "scatter", "bar chart", "line chart", "pie chart", "donut",
            "show as a chart", "show as chart", "show the data as a chart", "show data as a chart",
            "useful visualizations", "different useful visualizations", "generate visualizations",
            "create visualizations", "trend over time", "distribution of",
        )) or bool(re.search(r"\b(?:plot|chart|graph|visualize|visualise)\s+(?:the\s+)?(?:data|dataset|revenue|sales|salary|employees?|value|temperature|height|weight|[a-z0-9_]+)\b", q))

        has_explicit_ml_training = any(phrase in q for phrase in (
            "train a model", "train the model", "train a classifier", "train a regressor", "fit a model", "fit the model",
            "train classification", "train regression", "model accuracy",
        )) or bool(re.search(r"\b(?:train|retrain|fit)\s+(?:a\s+)?(?:new\s+)?(?:classification\s+|regression\s+)?(?:model|classifier|regressor)\b", q))

        has_explicit_ml_prediction = bool(re.search(r"\b(?:predict|predictions?|predicting|inference|score sample|evaluate sample)\b", q)) and not any(w in q for w in ("predicting a chart", "predict a chart"))

        if has_explicit_profiling:
            return RouteDecision(route="data", reason="Deterministic dataset profiling request", confidence=1.0)

        if has_explicit_viz and not (has_explicit_ml_training or has_explicit_ml_prediction):
            return RouteDecision(route="data", reason="Deterministic dataset visualization request", confidence=1.0)

        if has_explicit_ml_training or has_explicit_ml_prediction:
            return RouteDecision(route="ml", reason="Deterministic machine-learning request", confidence=1.0)

        # 4. Explicit personal facts and conversational identity (runs before document phrase matching)
        if any(q.startswith(greeting) for greeting in (
            "hello", "hi", "hey", "how are you", "good morning", "good afternoon", "good evening",
        )):
            return RouteDecision(route="general", reason="Deterministic conversational greeting", confidence=1.0)
        if any(phrase in q for phrase in ("what can you do", "what do you do", "your capabilities")):
            return RouteDecision(route="general", reason="Deterministic capability request", confidence=1.0)

        # Dog / pet statements and recall
        if re.search(r"\b(?:my\s+(?:dog|pet)(?:'s|s)?\s+name\s+is|my\s+(?:dog|pet)\s+is\s+named)\b", q):
            return RouteDecision(route="general", reason="Deterministic dog/pet name statement", confidence=1.0)
        if re.search(r"\b(?:what(?:'s|\s+is)|do\s+(?:you|u)\s+know|remember)\s+(?:my\s+)?(?:dog|pet)(?:'s|s)?\s+name\b", q):
            return RouteDecision(route="general", reason="Deterministic dog/pet name recall", confidence=1.0)

        # Project name statements and recall
        if re.search(r"\b(?:my\s+project(?:\s+name)?\s+(?:is|is\s+named)|project\s+is\s+named)\b", q):
            return RouteDecision(route="general", reason="Deterministic project name statement", confidence=1.0)
        if re.search(r"\b(?:what(?:'s|\s+is)|do\s+(?:you|u)\s+know|remember)\s+(?:my\s+)?project(?:'s)?\s+name\b", q):
            return RouteDecision(route="general", reason="Deterministic project name recall", confidence=1.0)

        # Personal identity statements and recall
        if re.search(r"\bmy name is\b|\bcall me\b|\bi am\b(?! sure|\s+not|\s+going|\s+trying|\s+looking|\s+interested|\s+here|\s+ready)|\bi'm\b(?!\s+not\b|\s+just\b|\s+using\b|\s+trying\b|\s+looking\b|\s+going\b|\s+a\b)", q):
            return RouteDecision(route="general", reason="Deterministic conversational identity statement", confidence=1.0)
        if re.search(r"\b(?:do\s+you|u)\s+know\s+my\s+name\b|\bwhat(?:'s|\s+is)\s+my\s+name\s*\??\s*$|\bwho\s+am\s+i\s*\??\s*$|\bremember\s+my\s+name\b|\bwhat\s+is\s+your\s+name\b|\bwho\s+are\s+you\b", q):
            return RouteDecision(route="general", reason="Deterministic conversational identity request", confidence=1.0)

        # 5. Explicit document / curriculum / section reference
        if has_documents and (
            re.search(r"\b(?:this|that|the|attached|uploaded|given)\s+(?:document|pdf|file|person|individual|guy|record|curriculum|syllabus|handbook|policy|manual|contract|agreement|paper|report|guide|text|material|course|section|chapter|part|topic|module|unit)\b", q)
            or re.search(r"\b(?:according to|in|from|about)\s+(?:the\s+)?(?:document|pdf|file|curriculum|syllabus|handbook|policy|manual|contract|paper|report|guide|section|chapter)\b", q)
            or any(phrase in q for phrase in (
                "what does it say", "list topics", "give me topics", "what are the topics",
                "what are the sections", "what are the chapters", "from the curriculum", "from the syllabus",
                "from the given curriculum", "in the curriculum", "in the syllabus", "in the handbook",
                "in the document", "in the pdf", "give me more", "more topics", "other topics",
                "additional topics", "next topics", "anything other", "anything else", "other than",
                "different topics", "remaining topics", "tell me more", "mentioned in that section",
                "in that section", "in this section", "that section", "this section", "those methods",
                "those technologies", "those tools", "technologies mentioned", "methods mentioned",
            ))
            or re.search(r"\b(?:more|other|additional|next|remaining)\s+(?:topics?|sections?|chapters?|modules?|points?)\b", q)
            or re.search(r"\b(?:technologies|methods|tools|algorithms|frameworks)\s+(?:mentioned|in|from)\b", q)
        ):
            return RouteDecision(route="rag", reason="Deterministic attached-document reference", confidence=1.0)

        return None

    def _heuristic_route(self, query: str, has_documents: bool = False) -> RouteDecision:
        """Deterministic keyword classification fallback."""
        q = query.lower().strip()
        if any(q.startswith(w) for w in ["hello", "hi", "hey", "how are you", "good morning", "good evening", "good afternoon"]):
            return RouteDecision(route="general", reason="Conversational greeting", confidence=1.0)
        if re.search(r"\b(?:dog|pet|project|my\s+name|who\s+am\s+i|your\s+name)\b", q):
            return RouteDecision(route="general", reason="Conversational personal context", confidence=1.0)
        if any(w in q for w in ["profile", "dataset", "columns", "rows", "csv", "tabular", "plot", "histogram", "scatter", "bar chart", "line chart", "pie chart", "donut", "chart"]):
            return RouteDecision(route="data", reason="Keyword match for Data profiling & visualization", confidence=0.8)
        if any(w in q for w in ["train", "predict", "classifier", "regressor", "accuracy", "model"]):
            return RouteDecision(route="ml", reason="Keyword match for Machine Learning", confidence=0.8)
        if any(w in q for w in ["image", "photo", "picture", "screenshot"]):
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
    turn_id = state.get("turn_id") or f"turn_{uuid.uuid4().hex[:8]}"

    return {
        "route": decision.route,
        "reason": decision.reason,
        "activities": activities,
        "turn_id": turn_id,
        "step_count": state.get("step_count", 0) + 1,
    }
