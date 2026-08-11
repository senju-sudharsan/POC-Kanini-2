from pydantic import BaseModel, Field


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
    """Phase 6 chat request supporting messages and optional image attachments."""

    messages: list[ChatMessage] = Field(min_length=1, max_length=40)
    attachments: list[ImageAttachment] = Field(default_factory=list, max_length=5)


class ChatResponse(BaseModel):
    """Phase 6 chat response."""

    message: ChatMessage
