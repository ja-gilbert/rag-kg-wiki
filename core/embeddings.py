from __future__ import annotations

import hashlib
import re
import warnings
from typing import Any, Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN = re.compile(r"[a-z0-9]+")

# Mirrors the `embedding:` block in config.yaml.
_DEFAULT_DIM = 256


def _tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A text with no usable tokens embeds to zero. Leave it there rather than
    # dividing by zero: its cosine against everything is then 0, which is the
    # honest answer.
    norms[norms == 0] = 1.0
    return matrix / norms


class HashEmbedder:
    """Deterministic bucket hashing with no semantics whatsoever.

    Here so the UI can show what retrieval looks like when the vector space
    carries no meaning at all: the machinery runs perfectly and ranks nonsense.
    """

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _tokenise(text):
                # hashlib rather than the builtin hash(): str hashing is salted
                # per process, so a saved index would stop matching fresh
                # queries on the next run.
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                vectors[i, int.from_bytes(digest, "big") % self.dim] += 1.0
        return _l2_normalise(vectors)


class TfidfSvdEmbedder:
    """LSA: tf-idf followed by truncated SVD. No download, works offline.

    Weaker than a trained sentence encoder and meant to look it -- it is the
    fallback that keeps the demo runnable on a machine with no network.
    """

    def __init__(self, corpus_texts: list[str], dim: int = _DEFAULT_DIM) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")
        matrix = self._vectorizer.fit_transform(corpus_texts)
        # SVD cannot yield more components than the smaller matrix dimension,
        # so a configured dim larger than the corpus is simply unreachable.
        # Clamp rather than crash, and expose what we actually got as .dim.
        self.dim = max(1, min(dim, min(matrix.shape) - 1))
        self._svd = TruncatedSVD(n_components=self.dim, random_state=0)
        self._svd.fit(matrix)

    def encode(self, texts: list[str]) -> np.ndarray:
        reduced = self._svd.transform(self._vectorizer.transform(texts))
        return _l2_normalise(reduced.astype(np.float32))


def _load_sentence_transformer(model: str, cache_dir: str | None) -> Any | None:
    """Construct the neural encoder, or return None if it is not installed.

    The import lives alone in this function so the fallback path can be tested
    by patching it, instead of by uninstalling a package mid-suite.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer(model, cache_folder=cache_dir)


class SentenceTransformerEmbedder:
    """The real one: a trained sentence encoder, all-MiniLM-L6-v2 by default."""

    def __init__(self, model: Any) -> None:
        self._model = model
        # Renamed in sentence-transformers 5.x; keep the old name working so the
        # pinned floor in requirements.txt (>=2.7) is not quietly a lie.
        get_dim = getattr(
            model, "get_embedding_dimension", None
        ) or model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        # Normalised here rather than trusting the model's own setting, so the
        # dot-product-is-cosine guarantee holds identically for all backends.
        return _l2_normalise(vectors.astype(np.float32))


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def build_embedder(
    cfg: dict[str, Any], corpus_texts: list[str] | None = None
) -> Embedder:
    backend = cfg["backend"]
    dim = cfg.get("dim", _DEFAULT_DIM)

    if backend == "hash":
        return HashEmbedder(dim)

    if backend == "tfidf-svd":
        return _build_tfidf_svd(corpus_texts, dim)

    if backend == "sentence-transformers":
        model = _load_sentence_transformer(
            cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            cfg.get("cache_dir"),
        )
        if model is None:
            warnings.warn(
                "sentence-transformers is not installed, falling back to the "
                "tfidf-svd backend. Install it, or set embedding.backend to "
                "tfidf-svd in config.yaml to silence this.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _build_tfidf_svd(corpus_texts, dim)
        return SentenceTransformerEmbedder(model)

    raise ValueError(f"unknown embedding backend {backend!r}")


def _build_tfidf_svd(corpus_texts: list[str] | None, dim: int) -> TfidfSvdEmbedder:
    if corpus_texts is None:
        raise ValueError(
            "the tfidf-svd backend is fitted on the corpus before use; "
            "pass corpus_texts= to build_embedder()"
        )
    return TfidfSvdEmbedder(corpus_texts, dim)
