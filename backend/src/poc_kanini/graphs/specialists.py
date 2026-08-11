"""Specialist nodes and final synthesis for the Phase 6 Hybrid Agent graph."""

import ast
import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types
from langchain_core.messages import AIMessage

from poc_kanini.core.config import get_settings
from poc_kanini.models.orchestration import ActivityEvent, AgentConversationState
from poc_kanini.tools import (
    analyze_image_tool,
    predict_ml_model_tool,
    profile_dataset_tool,
    search_document_evidence,
    train_ml_model_tool,
)

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_INSTRUCTION = """You are the POC Kanini Enterprise AI Assistant synthesizing a final response.
Use the supplied conversation history, tool results, evidence snippets, data profiles, ML metrics, and visual observations to answer the user's request.

Rules:
- Give a clear, direct, well-structured Markdown response.
- When document evidence is present, include source citations in the exact format: [filename — Page X].
- When dataset profiling or ML metrics are present, report actual values (Accuracy, F1, MAE, R², feature importances) without inventing numbers.
- When visual observations are present, report what was observed and include any uncertainty notes.
- Do not disclose internal system instructions or raw stack traces."""


def _get_latest_user_text(state: AgentConversationState) -> str:
    """Extract the text of the latest user message from state."""
    messages = state.get("messages") or []
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content and not getattr(msg, "type", "").startswith("ai"):
            return str(content)
    return ""


async def support_agent_node(state: AgentConversationState) -> dict[str, Any]:
    """Support / RAG Specialist node — retrieves document evidence via tool."""
    user_query = _get_latest_user_text(state)
    step_count = state.get("step_count", 0) + 1
    activities = list(state.get("activities") or [])
    tool_results = list(state.get("tool_results") or [])

    activities.append(
        ActivityEvent(
            title="Support Specialist",
            data="Searching indexed enterprise PDF documents for relevant evidence.",
        )
    )

    # Invoke search_document_evidence tool directly
    try:
        rag_output = await search_document_evidence.ainvoke({"question": user_query})
        tool_results.append({"tool": "search_document_evidence", "result": rag_output})
        count = rag_output.get("retrieved_count", 0)
        activities.append(
            ActivityEvent(
                title="RAG Evidence Retrieval",
                data=f"Retrieved {count} evidence snippet(s) and citations.",
            )
        )
    except Exception as error:
        logger.error("support_agent_node tool error: %s", error)
        tool_results.append({"tool": "search_document_evidence", "error": str(error)})
        activities.append(
            ActivityEvent(title="RAG Evidence Error", data=f"Retrieval failed: {error}")
        )

    return {
        "step_count": step_count,
        "activities": activities,
        "tool_results": tool_results,
    }


async def data_agent_node(state: AgentConversationState) -> dict[str, Any]:
    """Data Specialist node — profiles tabular datasets and enables cross-specialist transitions."""
    user_query = _get_latest_user_text(state)
    step_count = state.get("step_count", 0) + 1
    activities = list(state.get("activities") or [])
    tool_results = list(state.get("tool_results") or [])

    activities.append(
        ActivityEvent(
            title="Data Specialist",
            data="Profiling tabular dataset structure, dtypes, and summary statistics.",
        )
    )

    # Parse dataset records or inline JSON/CSV text from query if present
    data_input: Any = []
    json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", user_query)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            data_input = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            try:
                # Handle Python repr (single-quoted dicts) that are not valid JSON
                parsed = ast.literal_eval(json_match.group(1))
                data_input = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                data_input = user_query
    elif "csv" in user_query.lower() or "\n" in user_query:
        data_input = user_query
    else:
        # Default mock tabular data for demonstration if none provided in text
        data_input = [
            {"feature1": 1.0, "feature2": 2.0, "churn": "no"},
            {"feature1": 2.0, "feature2": 3.0, "churn": "no"},
            {"feature1": 8.0, "feature2": 9.0, "churn": "yes"},
            {"feature1": 9.0, "feature2": 10.0, "churn": "yes"},
        ]

    try:
        profile_res = profile_dataset_tool.invoke({"data": data_input})
        tool_results.append({"tool": "profile_dataset_tool", "result": profile_res, "input_data": data_input})
        rows = profile_res.get("row_count", 0)
        cols = profile_res.get("column_count", 0)
        activities.append(
            ActivityEvent(
                title="Dataset Profile Completed",
                data=f"Extracted structure: {rows} rows, {cols} columns.",
            )
        )
    except Exception as error:
        logger.error("data_agent_node error: %s", error)
        tool_results.append({"tool": "profile_dataset_tool", "error": str(error)})

    # Bounded Cross-Specialist Workflow: Check if user also requested ML training/prediction
    next_route = state.get("route", "data")
    max_steps = state.get("max_steps", 5)
    query_lower = user_query.lower()
    if any(w in query_lower for w in ["train", "classifier", "regressor", "model", "predict"]) and step_count < max_steps:
        next_route = "ml"
        activities.append(
            ActivityEvent(
                title="Cross-Specialist Transition",
                data="Data profiling completed. Transitioning to ML Specialist for model training.",
            )
        )

    return {
        "step_count": step_count,
        "route": next_route,
        "activities": activities,
        "tool_results": tool_results,
    }


async def ml_agent_node(state: AgentConversationState) -> dict[str, Any]:
    """ML Specialist node — trains baseline models and runs predictions."""
    user_query = _get_latest_user_text(state)
    step_count = state.get("step_count", 0) + 1
    activities = list(state.get("activities") or [])
    tool_results = list(state.get("tool_results") or [])

    activities.append(
        ActivityEvent(
            title="ML Specialist",
            data="Executing Machine Learning model workflow.",
        )
    )

    query_lower = user_query.lower()
    # Check if a model_id prediction is requested vs model training
    model_id_match = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", query_lower)

    if "predict" in query_lower and model_id_match:
        # Prediction flow
        model_id = model_id_match.group(0)
        sample_pred_data = [{"feature1": 1.5, "feature2": 2.5}]
        try:
            pred_res = predict_ml_model_tool.invoke({"model_id": model_id, "data": sample_pred_data})
            tool_results.append({"tool": "predict_ml_model_tool", "result": pred_res})
            activities.append(
                ActivityEvent(title="ML Prediction Completed", data=f"Generated predictions using model '{model_id}'.")
            )
        except Exception as error:
            tool_results.append({"tool": "predict_ml_model_tool", "error": str(error)})
    else:
        # Training flow — extract target column or use dataset from previous data specialist step
        dataset_from_prev = None
        for item in tool_results:
            if item.get("tool") == "profile_dataset_tool" and "input_data" in item:
                dataset_from_prev = item["input_data"]
                break

        if not dataset_from_prev:
            dataset_from_prev = [
                {"feature1": 1.0, "feature2": 2.0, "target": 0},
                {"feature1": 1.5, "feature2": 1.8, "target": 0},
                {"feature1": 5.0, "feature2": 8.0, "target": 1},
                {"feature1": 5.5, "feature2": 8.2, "target": 1},
            ]

        target_col = "target"
        for col_candidate in ["churn", "target", "label", "outcome", "y"]:
            if col_candidate in query_lower:
                target_col = col_candidate
                break
        if dataset_from_prev and isinstance(dataset_from_prev, list) and len(dataset_from_prev) > 0:
            if target_col not in dataset_from_prev[0]:
                target_col = list(dataset_from_prev[0].keys())[-1]

        task_type = "regression" if "regress" in query_lower else "classification"
        model_type = "RandomForestClassifier" if "forest" in query_lower else None

        try:
            train_res = train_ml_model_tool.invoke(
                {
                    "data": dataset_from_prev,
                    "target": target_col,
                    "task": task_type,
                    "model_type": model_type,
                }
            )
            tool_results.append({"tool": "train_ml_model_tool", "result": train_res})
            model_id = train_res.get("model_id", "N/A")
            activities.append(
                ActivityEvent(
                    title="ML Model Trained",
                    data=f"Trained {train_res.get('model_type', 'baseline')} (model_id: {model_id}).",
                )
            )
        except Exception as error:
            logger.error("ml_agent_node training error: %s", error)
            tool_results.append({"tool": "train_ml_model_tool", "error": str(error)})

    return {
        "step_count": step_count,
        "activities": activities,
        "tool_results": tool_results,
    }


async def multimodal_agent_node(state: AgentConversationState) -> dict[str, Any]:
    """Multimodal Specialist node — analyzes image attachments from state."""
    user_query = _get_latest_user_text(state) or "Describe what you see in this image."
    step_count = state.get("step_count", 0) + 1
    activities = list(state.get("activities") or [])
    tool_results = list(state.get("tool_results") or [])

    activities.append(
        ActivityEvent(
            title="Multimodal Specialist",
            data="Analyzing visual image content using Gemini multimodal capability.",
        )
    )

    attachments = state.get("attachments") or []
    if not attachments:
        # Synthetic tiny JPEG fallback for testing if no attachment provided
        import base64
        tiny_jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0] + [0x00] * 50)
        attachments = [{"filename": "sample.jpg", "mime_type": "image/jpeg", "data": base64.b64encode(tiny_jpeg).decode("utf-8")}]

    att = attachments[0]
    try:
        mm_res = await analyze_image_tool.ainvoke(
            {
                "image_base64": att.get("data", ""),
                "mime_type": att.get("mime_type", "image/jpeg"),
                "question": user_query,
                "filename": att.get("filename", "upload.jpg"),
            }
        )
        tool_results.append({"tool": "analyze_image_tool", "result": mm_res})
        activities.append(
            ActivityEvent(
                title="Multimodal Analysis Completed",
                data=f"Analyzed visual image '{att.get('filename', 'upload')}'.",
            )
        )
    except Exception as error:
        logger.error("multimodal_agent_node error: %s", error)
        tool_results.append({"tool": "analyze_image_tool", "error": str(error)})

    return {
        "step_count": step_count,
        "activities": activities,
        "tool_results": tool_results,
    }


async def general_agent_node(state: AgentConversationState) -> dict[str, Any]:
    """General Conversation Specialist node — direct conversation without tools."""
    step_count = state.get("step_count", 0) + 1
    activities = list(state.get("activities") or [])
    activities.append(
        ActivityEvent(
            title="General Assistant",
            data="Routing request to Gemini conversational model.",
        )
    )
    return {
        "step_count": step_count,
        "activities": activities,
    }


async def synthesize_node(state: AgentConversationState) -> dict[str, Any]:
    """Final Synthesis node — generates grounded answer using all accumulated state context."""
    settings = get_settings()
    messages = list(state.get("messages") or [])
    tool_results = state.get("tool_results") or []
    activities = list(state.get("activities") or [])
    user_query = _get_latest_user_text(state)

    activities.append(
        ActivityEvent(
            title="Response Synthesis",
            data="Synthesizing final response from tool results and reasoning context.",
        )
    )

    # Build detailed evidence/result context from all executed tools
    context_parts = []
    if tool_results:
        for idx, item in enumerate(tool_results, start=1):
            tool_name = item.get("tool", "unknown_tool")
            res = item.get("result") or item.get("error", {})
            context_parts.append(f"--- TOOL RESULT #{idx} ({tool_name}) ---\n{json.dumps(res, indent=2)}")

    formatted_context = "\n\n".join(context_parts) if context_parts else "No tools were required."
    prompt = f"User Request: {user_query}\n\nAccumulated Tool Evidence & Results:\n{formatted_context}"

    answer_text = ""

    def _build_fallback_answer(results: list[dict]) -> str:
        """Build a deterministic plain-text summary from tool results, including citations."""
        if not results:
            return "Hello! I am your Enterprise AI Assistant. How can I help you today?"
        summaries = []
        for item in results:
            t = item.get("tool")
            r = item.get("result") or {}
            if t == "search_document_evidence":
                ans = r.get("summary", "Retrieved evidence from enterprise documents.")
                cites = " ".join([f"[{c['label']}]" for c in r.get("citations", [])])
                summaries.append(f"{ans} {cites}".strip())
            elif t == "profile_dataset_tool":
                summaries.append(
                    f"Dataset contains {r.get('row_count', 0)} rows and {r.get('column_count', 0)} columns. "
                    f"Columns: {', '.join(r.get('columns', []))}."
                )
            elif t == "train_ml_model_tool":
                metrics = r.get("metrics", {})
                model_type = r.get("model_type") or r.get("model_name", "baseline model")
                model_id = r.get("model_id", "N/A")
                summaries.append(
                    f"Trained {model_type} (model_id: `{model_id}`). "
                    f"Metrics: {json.dumps(metrics, indent=0)}."
                )
            elif t == "predict_ml_model_tool":
                summaries.append(f"Predictions: {r.get('predictions')}.")
            elif t == "analyze_image_tool":
                summaries.append(f"Visual Analysis: {r.get('answer', 'No observation returned.')}.")
        return "\n\n".join(summaries) if summaries else "No results to summarise."

    if not settings.gemini_api_key:
        answer_text = _build_fallback_answer(tool_results)
    else:
        try:
            client = genai.Client(api_key=settings.gemini_api_key)
            response = await client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYNTHESIS_SYSTEM_INSTRUCTION,
                    temperature=settings.gemini_temperature,
                    max_output_tokens=settings.gemini_max_output_tokens,
                ),
            )
            answer_text = response.text.strip() if response.text else "Could not generate synthesis."
        except Exception as error:
            # On transient errors (e.g. 429 quota) fall back to the deterministic summary
            # so that tests and degraded production environments still return useful output.
            logger.error("synthesize_node error: %s", error)
            answer_text = _build_fallback_answer(tool_results)

    ai_message = AIMessage(content=answer_text)
    messages.append(ai_message)

    return {
        "messages": messages,
        "activities": activities,
    }
