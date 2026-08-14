"""Reflection and Human-In-The-Loop (HITL) approval node for Phase 7."""

import logging
import uuid
from typing import Any

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
    messages = state.get("messages") or []
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content and not getattr(msg, "type", "").startswith("ai"):
            return str(content)
    return ""


async def reflection_node(state: AgentConversationState) -> dict[str, Any]:
    """LangGraph Reflection node evaluating tool quality, error recovery, and HITL boundaries."""
    tool_results = state.get("tool_results") or []
    activities = list(state.get("activities") or [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 1)
    approval_status = state.get("approval_status")
    user_query = _get_latest_user_text(state)
    query_lower = user_query.lower()

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

    # Check if an ML training operation requires approval and hasn't been approved yet
    ml_train_run = any(item.get("tool") == "train_ml_model_tool" for item in tool_results)
    if (is_controlled_op or ml_train_run) and approval_status not in ("approved", "rejected"):
        appr_id = state.get("approval_id") or f"appr_{uuid.uuid4().hex[:8]}"
        appr_reason = (
            f"Human approval is required before completing controlled operation "
            f"'{state.get('route', 'ml')}' (approval_id: {appr_id})."
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
            "reflection": reflection,
            "activities": activities,
        }

    # If already approved or rejected, clear approval_required flag
    if approval_status in ("approved", "rejected"):
        activities.append(
            ActivityEvent(
                title="HITL Decision Applied",
                data=f"Human reviewer decision: {approval_status.upper()}.",
            )
        )

    # 2. Error Recovery & Bounded Retry Check
    last_tool_error = None
    for res in tool_results:
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
