"""Deterministic, reusable English text processing without external model downloads."""

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import spacy
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


@dataclass(frozen=True)
class TokenAnalysis:
    """Useful linguistic metadata for one token."""

    text: str
    normalized: str
    lemma: str
    is_alpha: bool
    is_stopword: bool
    is_punctuation: bool


class TextFileReader:
    """Read text files with explicit, predictable encoding fallback."""

    encodings = ("utf-8-sig", "utf-8", "utf-16", "latin-1")

    @classmethod
    def read(cls, path: str | Path) -> str:
        """Decode a text file, preserving Unicode text whenever possible."""

        source = Path(path)
        raw = source.read_bytes()
        for encoding in cls.encodings:
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError("unknown", raw, 0, len(raw), "No supported text encoding matched.")


class NlpProcessor:
    """Normalization, tokens, lemmas, stems, and semantic cleaning for English text."""

    def __init__(self, extra_stopwords: set[str] | None = None) -> None:
        self._nlp = spacy.blank("en")
        self._nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        self._nlp.initialize()
        self._stemmer = SnowballStemmer("english")
        self._stopwords = frozenset(ENGLISH_STOP_WORDS | {word.casefold() for word in extra_stopwords or set()})

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize Unicode, whitespace, and case while retaining meaningful characters."""

        normalized = unicodedata.normalize("NFKC", text).casefold()
        return re.sub(r"\s+", " ", normalized).strip()

    def tokenize(self, text: str) -> list[str]:
        """Tokenize normalized text with spaCy's language-aware tokenizer."""

        return [token.text for token in self._nlp.make_doc(self.normalize(text)) if not token.is_space]

    def analyze(self, text: str) -> list[TokenAnalysis]:
        """Return basic linguistic properties for each token."""

        doc = self._nlp(self.normalize(text))
        return [
            TokenAnalysis(
                text=token.text,
                normalized=token.lower_,
                lemma=token.lemma_.casefold(),
                is_alpha=token.is_alpha,
                is_stopword=token.lower_ in self._stopwords,
                is_punctuation=token.is_punct,
            )
            for token in doc
            if not token.is_space
        ]

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove configurable English stopwords without changing token order."""

        return [token for token in tokens if token.casefold() not in self._stopwords]

    def stem(self, tokens: list[str]) -> list[str]:
        """Stem alphabetic tokens with NLTK's Snowball stemmer."""

        return [self._stemmer.stem(token.casefold()) if token.isalpha() else token for token in tokens]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Lemmatize tokens with spaCy's packaged English lookup data."""

        doc = self._nlp(" ".join(tokens))
        return [token.lemma_.casefold() for token in doc]

    def semantic_clean(self, text: str, minimum_token_length: int = 2) -> list[str]:
        """Keep normalized, lemmatized content terms for downstream retrieval/classification."""

        analysis = self.analyze(text)
        return [
            token.lemma
            for token in analysis
            if token.is_alpha and not token.is_stopword and len(token.lemma) >= minimum_token_length
        ]
