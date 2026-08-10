"""Optional extension point for later Gemini page/image understanding."""

from typing import Protocol

from poc_kanini.models.documents import ProcessedPage


class PageUnderstandingHook(Protocol):
    """Later adapters may enrich visually complex pages without changing extraction contracts."""

    async def analyze_page(self, page: ProcessedPage) -> dict[str, str]: ...
