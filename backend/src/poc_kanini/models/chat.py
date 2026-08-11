from pydantic import BaseModel, Field
from poc_kanini.models.orchestration import ActivityEvent


class ImageAttachment(BaseModel):
    """An optional base64-encoded image attached to a chat request."""

    filename: str = Field(default="upload.png")
    mime_type: str = Field(default="image/jpeg", pattern="^image/(jpeg|png|webp)$")
    data: str = Field(min_length=1, description="Base64-encoded image bytes.")


class ChatMessage(BaseModel):
    """A validated browser/API message used to build LangGraph state."""

    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    """Phase 7 chat request supporting messages, attachments, thread persistence, and approval decisions."""

    messages: list[ChatMessage] = Field(min_length=1, max_length=40)
    attachments: list[ImageAttachment] = Field(default_factory=list, max_length=5)
    thread_id: str | None = Field(default=None, description="Optional thread/session identifier for checkpointed context")
    approval: str | None = Field(default=None, pattern="^(approved|rejected)$", description="Human approval decision for interrupted operations")


class ChatResponse(BaseModel):
    """Phase 7 chat response returning assistant message, thread_id, activities, and optional approval request."""

    message: ChatMessage
    thread_id: str = Field(description="Thread/session ID associated with the checkpointed state")
    approval_required: bool = Field(default=False, description="True if graph execution was interrupted awaiting human approval")
    approval_id: str | None = Field(default=None, description="Unique identifier for the approval request")
    approval_reason: str | None = Field(default=None, description="Human-readable explanation of why approval is required")
    activities: list[ActivityEvent] = Field(default_factory=list, description="Activities emitted during execution")
