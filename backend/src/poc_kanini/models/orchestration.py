from typing import Annotated, Literal

from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ActivityEvent(TypedDict):
    """A UI-safe status update emitted by a LangGraph node or tool."""

    title: str
    data: str


class ConversationState(TypedDict, total=False):
    """Shared conversation state for the Phase 1 Gemini chat graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    activities: list[ActivityEvent]
    thread_id: str


AgentKind = Literal["support", "data", "ml"]

RouteType = Literal["rag", "data", "ml", "multimodal", "general"]


class RouteDecision(BaseModel):
    """Structured supervisor routing decision."""

    route: RouteType = Field(default="general")
    reason: str = Field(default="")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class AgentConversationState(TypedDict, total=False):
    """Phase 6 full hybrid agent state extending ConversationState."""

    # Core LangGraph message accumulator
    messages: Annotated[list[AnyMessage], add_messages]

    # Activity feed for the UI status panel
    activities: list[ActivityEvent]

    # Optional image attachments: list of dicts with filename, mime_type, data (base64)
    attachments: list[dict]

    # Tool execution outputs accumulated across specialist nodes
    tool_results: list[dict]

    # Supervisor routing decision
    route: RouteType
    reason: str

    # Bounded execution counter
    step_count: int
    max_steps: int

    # Error tracking
    error: str | None

    # Conversation tracking
    thread_id: str
