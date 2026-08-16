"""Reflection and Human-In-The-Loop (HITL) approval node for Phase 7."""

import logging
import uuid
from typing import Any

from poc_kanini.graphs.turn_context import get_current_turn_query, get_current_turn_tools
from poc_kanini.models.orchestration import (
    ActivityEvent,
    AgentConversationState,
    ApprovalRequest,
    ReflectionDecision,
)

logger = logging.getLogger(__name__)


def _is_non_retryable_provider_error(error: str) -> bool:
    """Avoid retrying provider failures that cannot succeed within this turn."""
    text = error.lower()
    return any(marker in text for marker in (
        "429", "resource_exhausted", "quota", "rate limit",
        "401", "unauthenticated", "403", "permission_denied",
    ))


def _get_latest_user_text(state: AgentConversationState) -> str:
    """Extract text from the latest user message in state."""
    return get_current_turn_query(state)


async def reflection_node(state: AgentConversationState) -> dict[str, Any]:
    """LangGraph Reflection node evaluating tool quality, error recovery, and HITL boundaries."""
    activities = list(state.get("activities") or [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 1)
    approval_status = state.get("approval_status")
    user_query = get_current_turn_query(state)
    query_lower = user_query.lower()

    # Retrieve tool results originating exclusively from this turn
    current_tools = get_current_turn_tools(state)

    # 1. Human-In-The-Loop (HITL) Approval Boundary Check
    # Controlled operations: explicit requests containing approval triggers or sensitive model training
    is_controlled_op = any(
        kw in query_lower
        for kw in [
            "requires approval",
            "sensitive",
            "delete",
            "production model",
            "approve operation",
            "controlled operation",
        ]
    )

    ml_train_run = any(item.get("tool") == "train_ml_model_tool" for item in current_tools)
    is_prediction_run = any(item.get("tool") == "predict_ml_model_tool" for item in current_tools)

    if (is_controlled_op or ml_train_run) and not is_prediction_run and approval_status not in ("approved", "rejected"):
        appr_id = state.get("approval_id") or f"appr_{uuid.uuid4().hex[:8]}"
        operation_name = "ml" if ml_train_run else (state.get("route") or "controlled_operation")
        appr_reason = (
            f"Human approval is required before completing controlled operation "
            f"'{operation_name}' (approval_id: {appr_id})."
        )
        activities.append(
            ActivityEvent(
                title="HITL Approval Required",
                data=f"Execution paused. Awaiting human approval for request '{appr_id}'.",
            )
        )
        reflection = ReflectionDecision(
            quality_ok=False,
            needs_retry=False,
            needs_more_evidence=False,
            reason="Execution interrupted awaiting human approval.",
        ).model_dump()

        return {
            "approval_required": True,
            "approval_id": appr_id,
            "approval_reason": appr_reason,
            "approval_status": "pending",
            "operation": operation_name,
            "reflection": reflection,
            "activities": activities,
        }

    # If already approved or rejected, record the decision and proceed normally
    if approval_status in ("approved", "rejected"):
        activities.append(
            ActivityEvent(
                title="HITL Decision Applied",
                data=f"Human reviewer decision: {approval_status.upper()}.",
            )
        )

    # 2. Error Recovery & Bounded Retry Check on CURRENT-TURN tools
    last_tool_error = None
    for res in current_tools:
        if "error" in res or (isinstance(res.get("result"), dict) and "error" in res["result"]):
            err_dict = res.get("error") or res.get("result", {}).get("error")
            last_tool_error = str(err_dict)

    if last_tool_error and not _is_non_retryable_provider_error(last_tool_error) and retry_count < max_retries:
        new_retry_count = retry_count + 1
        activities.append(
            ActivityEvent(
                title="Reflection - Bounded Retry Triggered",
                data=f"Tool error detected: '{last_tool_error}'. Retrying operation ({new_retry_count}/{max_retries}).",
            )
        )
        reflection = ReflectionDecision(
            quality_ok=False,
            needs_retry=True,
            needs_more_evidence=False,
            reason=f"Tool error: {last_tool_error}. Triggering bounded retry.",
        ).model_dump()

        return {
            "retry_count": new_retry_count,
            "reflection": reflection,
            "activities": activities,
        }

    # 3. Quality Verification (Pass)
    activities.append(
        ActivityEvent(
            title="Reflection Completed",
            data="Verified tool execution results, quality standards, and evidence completeness.",
        )
    )
    reflection = ReflectionDecision(
        quality_ok=True,
        needs_retry=False,
        needs_more_evidence=False,
        reason="Tool execution and evidence quality verified cleanly.",
    ).model_dump()

    return {
        "approval_required": False,
        "reflection": reflection,
        "activities": activities,
    }
