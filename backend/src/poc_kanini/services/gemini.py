"""Gemini adapter kept separate from graph orchestration and API concerns."""

from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from poc_kanini.core.config import Settings


SYSTEM_INSTRUCTION = """You are the POC Kanini Enterprise AI Assistant.
Help users reason about general enterprise questions using clear, accurate text.
At this stage you have no access to uploaded documents, enterprise databases,
external web search, data-analysis tools, or ML prediction tools. Say so plainly
when a request needs one of those planned capabilities. Do not invent evidence,
citations, actions, or data. Be concise, practical, and transparent about limits."""


def _contents_from_messages(messages: list[AnyMessage]) -> list[types.Content]:
    """Convert LangGraph messages to the Gemini SDK's explicit content format."""

    contents: list[types.Content] = []
    for message in messages:
        content = message.content if isinstance(message.content, str) else str(message.content)
        role = "model" if isinstance(message, AIMessage) else "user"
        if isinstance(message, (HumanMessage, AIMessage)):
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=content)]))
    return contents


class GeminiChatService:
    """Generate a text response from Gemini for the Phase 1 chat graph."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def respond(self, messages: list[AnyMessage]) -> str:
        """Generate a grounded, text-only response using the configured Gemini model."""

        if not self._settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        client = genai.Client(api_key=self._settings.gemini_api_key)
        response = await client.aio.models.generate_content(
            model=self._settings.gemini_model,
            contents=_contents_from_messages(messages),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=self._settings.gemini_temperature,
                max_output_tokens=self._settings.gemini_max_output_tokens,
            ),
        )
        return response.text.strip() if response.text else "I could not generate a text response."
