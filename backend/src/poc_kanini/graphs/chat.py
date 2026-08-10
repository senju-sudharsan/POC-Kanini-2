"""The deliberately small, text-only LangGraph workflow for Phase 1."""

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from poc_kanini.core.config import get_settings
from poc_kanini.models.orchestration import ConversationState
from poc_kanini.services.gemini import GeminiChatService


async def generate_answer(state: ConversationState) -> dict[str, list[AIMessage]]:
    """Ask Gemini to answer the current conversation."""

    answer = await GeminiChatService(get_settings()).respond(state["messages"])
    return {"messages": [AIMessage(content=answer)]}


builder = StateGraph(ConversationState)
builder.add_node("generate_answer", generate_answer)
builder.add_edge(START, "generate_answer")
builder.add_edge("generate_answer", END)
chat_graph = builder.compile(name="phase-1-gemini-chat")
