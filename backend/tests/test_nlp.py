from pathlib import Path

import pytest

from poc_kanini.nlp.embeddings import TfidfEmbeddingModel, WordEmbeddingModel
from poc_kanini.nlp.processor import NlpProcessor, TextFileReader


@pytest.fixture()
def processor() -> NlpProcessor:
    return NlpProcessor(extra_stopwords={"kanini"})


def test_reads_utf8_sig_text_file(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_bytes("Résumé for Kanini".encode("utf-8-sig"))
    assert TextFileReader.read(source) == "Résumé for Kanini"


def test_normalizes_unicode_case_and_whitespace(processor: NlpProcessor) -> None:
    assert processor.normalize("  Café\u00a0\u212aANINI\n") == "café kanini"


def test_tokenization_and_linguistic_analysis(processor: NlpProcessor) -> None:
    assert processor.tokenize("Hello, Kanini!") == ["hello", ",", "kanini", "!"]
    analysis = processor.analyze("Running reports.")
    assert analysis[0].is_alpha and analysis[0].lemma == "run"
    assert analysis[-1].is_punctuation


def test_stopwords_stemming_and_lemmatization(processor: NlpProcessor) -> None:
    assert processor.remove_stopwords(["the", "enterprise", "and", "kanini"]) == ["enterprise"]
    assert processor.stem(["running", "reports"]) == ["run", "report"]
    assert processor.lemmatize(["mice", "running", "reports"]) == ["mouse", "run", "report"]


def test_semantic_cleaning_removes_noise_and_keeps_content(processor: NlpProcessor) -> None:
    assert processor.semantic_clean("The 2026 reports are running for Kanini!") == ["report", "run"]


def test_embeddings_and_similarity() -> None:
    documents = ["revenue and profit increased", "profit and revenue improved", "weather forecast rain"]
    document_embeddings = TfidfEmbeddingModel().fit_transform(documents)
    assert document_embeddings.shape[0] == 3
    assert TfidfEmbeddingModel.similarity(document_embeddings[0], document_embeddings[1]) > TfidfEmbeddingModel.similarity(document_embeddings[0], document_embeddings[2])

    word_embeddings = WordEmbeddingModel().fit(documents)
    assert word_embeddings.vector_for("revenue").size == 3
    assert word_embeddings.similarity("revenue", "profit") > word_embeddings.similarity("revenue", "weather")
