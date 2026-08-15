"""Turn Context and Provenance Utilities for AURA.

Provides a unified boundary between long-lived conversational memory
and current-turn execution state (tool results, citations, reports, warnings).
"""

from typing import Any
from langchain_core.messages import AnyMessage


def get_current_turn_query(state: dict[str, Any]) -> str:
    """Extract the latest user query from conversation state."""
    messages = state.get("messages") or []
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content and not getattr(msg, "type", "").startswith("ai") and getattr(msg, "role", "") != "assistant":
            return str(content).strip()
    return ""


def get_current_turn_id(state: dict[str, Any]) -> str:
    """Get or generate the identifier for the current turn."""
    turn_id = state.get("turn_id")
    if turn_id:
        return str(turn_id)
    # Fallback to hash of latest user text and step count
    query = get_current_turn_query(state)
    step = state.get("step_count", 0)
    return f"turn_{step}_{abs(hash(query)) % 1000000:06d}"


def tag_tool_result(
    tool_name: str,
    state: dict[str, Any],
    result: Any = None,
    error: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a standardized, turn-tagged tool result dictionary."""
    turn_id = get_current_turn_id(state)
    query = get_current_turn_query(state)
    record: dict[str, Any] = {
        "tool": tool_name,
        "turn_id": turn_id,
        "query": query,
    }
    if result is not None:
        record["result"] = result
    if error is not None:
        record["error"] = str(error)
    if extra:
        record.update(extra)
    return record


def get_current_turn_tools(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Retrieve tool execution results strictly originating from the current turn."""
    all_tools = state.get("tool_results") or []
    if not all_tools:
        return []

    turn_id = state.get("turn_id")
    query = get_current_turn_query(state)

    # If no tool in all_tools has turn metadata (e.g. direct test fixture), return all
    has_tagged_tools = any("turn_id" in item or "query" in item for item in all_tools)
    if not has_tagged_tools:
        return list(all_tools)

    current_tools: list[dict[str, Any]] = []
    for item in all_tools:
        # Match by explicit turn_id if available
        if turn_id and item.get("turn_id") == turn_id:
            current_tools.append(item)
        # Match by query if turn_id not set
        elif query and item.get("query") == query:
            current_tools.append(item)
        elif not item.get("turn_id") and not item.get("query"):
            current_tools.append(item)

    return current_tools


def get_current_turn_rag_results(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Get current turn's document RAG retrieval results."""
    return [t for t in get_current_turn_tools(state) if t.get("tool") == "search_document_evidence"]


def get_current_turn_ml_results(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Get current turn's ML training and prediction tool results."""
    return [t for t in get_current_turn_tools(state) if t.get("tool") in ("train_ml_model_tool", "predict_ml_model_tool")]


def get_current_turn_data_results(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Get current turn's dataset profiling tool results."""
    return [t for t in get_current_turn_tools(state) if t.get("tool") == "profile_dataset_tool"]


def get_current_turn_multimodal_results(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Get current turn's visual image analysis tool results."""
    return [t for t in get_current_turn_tools(state) if t.get("tool") == "analyze_image_tool"]
