"""Phase 7 Stateful Hybrid Agent StateGraph assembly with Reflection, HITL, and Checkpointing."""

import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from poc_kanini.graphs.reflection import reflection_node
from poc_kanini.graphs.specialists import (
    data_agent_node,
    general_agent_node,
    ml_agent_node,
    multimodal_agent_node,
    support_agent_node,
    synthesize_node,
)
from poc_kanini.graphs.supervisor import supervisor_node
from poc_kanini.models.orchestration import AgentConversationState

logger = logging.getLogger(__name__)


def route_supervisor_decision(state: AgentConversationState) -> str:
    """Route from supervisor to the selected specialist node."""
    route = state.get("route", "general")
    if route in ("rag", "support"):
        return "support_agent"
    elif route == "data":
        return "data_agent"
    elif route == "ml":
        return "ml_agent"
    elif route == "multimodal":
        return "multimodal_agent"
    return "general_agent"


def route_reflection_decision(state: AgentConversationState) -> str:
    """Determine routing after reflection: retry, cross-specialist, approval pause, or synthesis."""
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 5)

    if step_count >= max_steps:
        return "synthesize"

    # 1. Pause for Human Approval if required and pending
    if state.get("approval_required") and state.get("approval_status") not in ("approved", "rejected"):
        return "synthesize"

    reflection = state.get("reflection") or {}

    # 2. Bounded Retry Routing
    if reflection.get("needs_retry"):
        return route_supervisor_decision(state)

    # 3. Cross-Specialist Routing (Data -> ML)
    route = state.get("route", "")
    if route == "ml":
        from poc_kanini.graphs.turn_context import get_current_turn_tools
        current_tools = get_current_turn_tools(state)
        ml_tool_run = any(item.get("tool") in ("train_ml_model_tool", "predict_ml_model_tool") for item in current_tools)
        if not ml_tool_run:
            return "ml_agent"

    return "synthesize"


# Build the flat Hybrid Agent StateGraph
builder = StateGraph(AgentConversationState)

# Add all graph nodes
builder.add_node("supervisor", supervisor_node)
builder.add_node("support_agent", support_agent_node)
builder.add_node("data_agent", data_agent_node)
builder.add_node("ml_agent", ml_agent_node)
builder.add_node("multimodal_agent", multimodal_agent_node)
builder.add_node("general_agent", general_agent_node)
builder.add_node("reflection", reflection_node)
builder.add_node("synthesize", synthesize_node)

# Add edges
builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_supervisor_decision,
    {
        "support_agent": "support_agent",
        "data_agent": "data_agent",
        "ml_agent": "ml_agent",
        "multimodal_agent": "multimodal_agent",
        "general_agent": "general_agent",
    },
)

# All specialist nodes proceed to reflection node
builder.add_edge("support_agent", "reflection")
builder.add_edge("data_agent", "reflection")
builder.add_edge("ml_agent", "reflection")
builder.add_edge("multimodal_agent", "reflection")
builder.add_edge("general_agent", "reflection")

# Reflection node routes to retry specialist, cross-specialist, or synthesis
builder.add_conditional_edges(
    "reflection",
    route_reflection_decision,
    {
        "support_agent": "support_agent",
        "data_agent": "data_agent",
        "ml_agent": "ml_agent",
        "multimodal_agent": "multimodal_agent",
        "synthesize": "synthesize",
    },
)

builder.add_edge("synthesize", END)

# In-memory checkpointer for thread-isolated state persistence
memory_checkpointer = MemorySaver()

# Compile stateful graph with checkpointer
hybrid_chat_graph = builder.compile(
    checkpointer=memory_checkpointer,
    name="phase-7-stateful-hybrid-agent",
)

# Backwards compatibility export
chat_graph = hybrid_chat_graph
