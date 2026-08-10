"""Small, deterministic layout heuristics suitable for a document-processing POC."""

import re

from poc_kanini.models.documents import DocumentStructure


def analyze_layout(text: str) -> DocumentStructure:
    """Identify headings, bullets, table-like rows, and paragraphs without a heavy layout model."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    headings: list[str] = []
    list_items: list[str] = []
    table_lines: list[str] = []
    paragraphs: list[str] = []
    for line in lines:
        if re.match(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?[A-Z][A-Z\s]{3,}$", line) or line.endswith(":"):
            headings.append(line)
        elif re.match(r"^(?:[-*•]|\d+[.)])\s+", line):
            list_items.append(line)
        elif "|" in line or re.search(r"\S\s{3,}\S", line):
            table_lines.append(line)
        else:
            paragraphs.append(line)
    return DocumentStructure(headings=headings, paragraphs=paragraphs, list_items=list_items, table_lines=table_lines)
