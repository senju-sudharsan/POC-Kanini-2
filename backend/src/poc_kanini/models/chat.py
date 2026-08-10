from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A validated browser/API message used to build LangGraph state."""

    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    """Phase 1 text-only Gemini chat request."""

    messages: list[ChatMessage] = Field(min_length=1, max_length=40)


class ChatResponse(BaseModel):
    """Phase 1 text-only Gemini chat response."""

    message: ChatMessage
