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


class ReflectionDecision(BaseModel):
    """Structured reflection outcome evaluating tool results and answer quality."""

    quality_ok: bool = Field(default=True, description="True if evidence and tool execution quality are sufficient.")
    needs_retry: bool = Field(default=False, description="True if a recoverable tool error occurred and retry budget remains.")
    needs_more_evidence: bool = Field(default=False, description="True if retrieved evidence is incomplete.")
    reason: str = Field(default="", description="Explanation for the reflection assessment.")


class ApprovalRequest(BaseModel):
    """Structured human-in-the-loop approval details."""

    approval_required: bool = Field(default=False, description="True if graph execution was interrupted awaiting human approval.")
    approval_id: str | None = Field(default=None, description="Unique identifier for the approval request.")
    approval_reason: str | None = Field(default=None, description="Human-readable explanation of why approval is required.")
    operation: str | None = Field(default=None, description="The specific bounded operation requiring approval.")


class AgentConversationState(TypedDict, total=False):
    """Phase 7 full stateful hybrid agent state extending ConversationState."""

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

    # Bounded execution counters
    step_count: int
    max_steps: int
    retry_count: int
    max_retries: int

    # Phase 7 Reflection
    reflection: dict

    # Phase 7 Human-In-The-Loop Approval
    approval_required: bool
    approval_id: str | None
    approval_reason: str | None
    approval_status: str | None  # "pending", "approved", "rejected"

    # Error tracking
    error: str | None

    # Phase 8 Extensions: Document association, citations, reports, actions, warnings
    document_ids: list[str]
    citations: list[dict]
    reports: list[dict]
    actions: list[dict]
    warnings: list[str]
    synthesis_status: str

    # Thread/session tracking
    thread_id: str
