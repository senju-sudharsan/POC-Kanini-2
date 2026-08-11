"""Phase 6 Hybrid Agent StateGraph assembly and compilation."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

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


def route_specialist_transition(state: AgentConversationState) -> str:
    """Determine whether to transition to another specialist (e.g. Data -> ML) or synthesize."""
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 5)

    if step_count >= max_steps:
        return "synthesize"

    route = state.get("route", "")
    # Handle cross-specialist Data -> ML transition
    if route == "ml":
        # Check if we haven't already run ML agent twice
        tool_results = state.get("tool_results") or []
        ml_tool_run = any(item.get("tool") in ("train_ml_model_tool", "predict_ml_model_tool") for item in tool_results)
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

builder.add_conditional_edges(
    "support_agent",
    route_specialist_transition,
    {"ml_agent": "ml_agent", "synthesize": "synthesize"},
)

builder.add_conditional_edges(
    "data_agent",
    route_specialist_transition,
    {"ml_agent": "ml_agent", "synthesize": "synthesize"},
)

builder.add_conditional_edges(
    "ml_agent",
    route_specialist_transition,
    {"ml_agent": "ml_agent", "synthesize": "synthesize"},
)

builder.add_conditional_edges(
    "multimodal_agent",
    route_specialist_transition,
    {"ml_agent": "ml_agent", "synthesize": "synthesize"},
)

builder.add_edge("general_agent", "synthesize")
builder.add_edge("synthesize", END)

# Compile the graph
hybrid_chat_graph = builder.compile(name="phase-6-hybrid-agent")

# Backwards compatibility export for chat endpoint
chat_graph = hybrid_chat_graph
