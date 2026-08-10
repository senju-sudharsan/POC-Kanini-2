from typing import Annotated, Literal

from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict


class ActivityEvent(TypedDict):
    """A UI-safe status update emitted by a future LangGraph node or tool."""

    title: str
    data: str


class ConversationState(TypedDict, total=False):
    """Shared conversation state for the Phase 1 Gemini chat graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    activities: list[ActivityEvent]
    thread_id: str


AgentKind = Literal["support", "data", "ml"]
