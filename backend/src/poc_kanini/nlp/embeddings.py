"""Local embeddings and similarity primitives that require no downloaded model."""

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfEmbeddingModel:
    """Create reusable dense text embeddings from a fitted TF-IDF representation."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
        self._fitted = False

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        """Fit the representation and return one embedding vector per text."""

        self._fitted = True
        return self._vectorizer.fit_transform(texts).toarray()

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Embed additional texts using the existing fitted vocabulary."""

        if not self._fitted:
            raise RuntimeError("Fit the embedding model before transforming text.")
        return self._vectorizer.transform(texts).toarray()

    @staticmethod
    def similarity(left: np.ndarray, right: np.ndarray) -> float:
        """Calculate cosine similarity for two individual embedding vectors."""

        return float(cosine_similarity(left.reshape(1, -1), right.reshape(1, -1))[0, 0])


class WordEmbeddingModel:
    """Create corpus-fitted word embeddings from words' TF-IDF document contexts."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self._word_vectors: dict[str, np.ndarray] = {}

    def fit(self, texts: Sequence[str]) -> "WordEmbeddingModel":
        """Fit word context vectors from a corpus without downloading a pretrained model."""

        matrix = self._vectorizer.fit_transform(texts)
        terms = self._vectorizer.get_feature_names_out()
        self._word_vectors = {term: matrix[:, index].toarray().ravel() for index, term in enumerate(terms)}
        return self

    def vector_for(self, word: str) -> np.ndarray:
        """Return an embedding for a known word."""

        try:
            return self._word_vectors[word.casefold()]
        except KeyError as error:
            raise KeyError(f"No embedding is available for {word!r}; fit a corpus containing the word first.") from error

    def similarity(self, left: str, right: str) -> float:
        """Calculate cosine similarity between two corpus-fitted word embeddings."""

        return TfidfEmbeddingModel.similarity(self.vector_for(left), self.vector_for(right))
