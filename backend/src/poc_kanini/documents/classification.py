"""Transparent, lightweight document categorisation for Phase 2."""

from poc_kanini.models.documents import DocumentCategory


KEYWORDS: dict[DocumentCategory, tuple[str, ...]] = {
    "policy": ("policy", "compliance", "effective date", "shall"),
    "report": ("report", "executive summary", "findings", "quarterly"),
    "handbook": ("handbook", "employee", "code of conduct", "benefits"),
    "guideline": ("guideline", "guidelines", "recommended", "best practice"),
    "other": (),
}


def classify_document(text: str) -> DocumentCategory:
    """Select the category with the most matching indicator terms; ties remain other."""

    normalized = text.casefold()
    scores = {category: sum(keyword in normalized for keyword in keywords) for category, keywords in KEYWORDS.items() if keywords}
    category, score = max(scores.items(), key=lambda item: item[1], default=("other", 0))
    return category if score else "other"
