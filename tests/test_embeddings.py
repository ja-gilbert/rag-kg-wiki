from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import embeddings
from core.chunking import chunk_documents
from core.corpus import load_corpus
from core.embeddings import TfidfSvdEmbedder, build_embedder

ROOT = Path(__file__).resolve().parent.parent


def _corpus_texts() -> list[str]:
    corpus = load_corpus(ROOT / "data" / "raw")
    return [c.text for c in chunk_documents(corpus, {"strategy": "paragraph"})]


def test_hash_embedder_returns_l2_normalised_vectors():
    embedder = build_embedder({"backend": "hash", "dim": 64})
    vectors = embedder.encode(
        ["Atlas depends on Ember.", "Beacon aggregates telemetry."]
    )
    assert vectors.shape == (2, 64)
    # L2-normalised is what lets the vector store use a plain dot product as
    # cosine similarity, so it is a property of every backend, not an extra.
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_tfidf_svd_embedder_returns_l2_normalised_vectors():
    embedder = build_embedder(
        {"backend": "tfidf-svd", "dim": 32}, corpus_texts=_corpus_texts()
    )
    vectors = embedder.encode(["Atlas depends on Ember for authentication."])
    assert vectors.shape == (1, 32)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_tfidf_svd_requires_the_corpus_it_must_be_fitted_on():
    # The asymmetry the spec calls out: LSA has to see the corpus first, the
    # neural backend must not. The factory hides it, but omitting the corpus
    # should say so plainly rather than fail somewhere inside sklearn.
    with pytest.raises(ValueError) as excinfo:
        build_embedder({"backend": "tfidf-svd", "dim": 32})
    assert "corpus_texts" in str(excinfo.value)


def test_missing_sentence_transformers_warns_and_falls_back_to_lsa(monkeypatch):
    # The demo has to run on a machine with no network, so an absent neural
    # backend degrades rather than crashes. Simulating absence beats uninstalling
    # it, and keeps this test meaningful whether or not it is installed here.
    monkeypatch.setattr(embeddings, "_load_sentence_transformer", lambda *a, **k: None)
    with pytest.warns(RuntimeWarning, match="tfidf-svd"):
        embedder = build_embedder(
            {"backend": "sentence-transformers", "model": "irrelevant", "dim": 32},
            corpus_texts=_corpus_texts(),
        )
    assert isinstance(embedder, TfidfSvdEmbedder)
    assert np.allclose(np.linalg.norm(embedder.encode(["Atlas."]), axis=1), 1.0)
