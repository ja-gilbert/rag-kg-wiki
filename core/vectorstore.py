from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from core.chunking import Chunk

_VECTORS_FILE = "vectors.npy"
_CHUNKS_FILE = "chunks.json"

# BM25 tokenises for itself rather than borrowing the embedder's tokeniser:
# lexical matching and vector hashing are free to diverge, and coupling them
# would make a change to one silently move the other.
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    score: float


class VectorStore:
    """Exact cosine over a dense matrix -- the whole search is one matmul.

    No FAISS and no approximate index: at a few hundred chunks an ANN structure
    would be a lie about where this architecture's complexity actually lives.
    """

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"{len(chunks)} chunks but {vectors.shape[0]} vectors; "
                "the index and the corpus have drifted apart"
            )
        self.chunks = chunks
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        # Both sides are L2-normalised, so the dot product is cosine similarity.
        scores = self.vectors @ query_vector
        # Stable sort so equal scores keep corpus order and results are
        # reproducible run to run.
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [self._hit(int(i), float(scores[i])) for i in order]

    def save(self, path: Path | str) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / _VECTORS_FILE, self.vectors)
        payload = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "text": c.text,
            }
            for c in self.chunks
        ]
        (directory / _CHUNKS_FILE).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> VectorStore:
        directory = Path(path)
        vectors_path = directory / _VECTORS_FILE
        chunks_path = directory / _CHUNKS_FILE
        if not (vectors_path.exists() and chunks_path.exists()):
            raise FileNotFoundError(
                f"no vector index at {directory}. "
                "Run `python -m scripts.build_all` first."
            )
        chunks = [
            Chunk(**row) for row in json.loads(chunks_path.read_text(encoding="utf-8"))
        ]
        return cls(chunks, np.load(vectors_path))

    def _hit(self, index: int, score: float) -> SearchHit:
        return _hit(self.chunks[index], score)


def _hit(chunk: Chunk, score: float) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        doc_title=chunk.doc_title,
        text=chunk.text,
        score=score,
    )


class BM25Index:
    """Okapi BM25, the lexical counterpart to the vector store.

    Kept beside the semantic index so the UI can put the two rankings for one
    query side by side -- which is where you can see that they disagree.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        tokenised = [_tokenise(c.text) for c in chunks]
        self._term_freqs = [Counter(t) for t in tokenised]
        self._lengths = np.array([len(t) for t in tokenised], dtype=np.float32)
        self._avg_len = float(self._lengths.mean()) if len(chunks) else 0.0

        document_freq: Counter[str] = Counter()
        for tokens in tokenised:
            document_freq.update(set(tokens))
        n = len(chunks)
        self._idf = {
            term: math.log(1.0 + (n - count + 0.5) / (count + 0.5))
            for term, count in document_freq.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        terms = _tokenise(query)
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for i, freqs in enumerate(self._term_freqs):
            if self._avg_len:
                norm = self.k1 * (
                    1 - self.b + self.b * float(self._lengths[i]) / self._avg_len
                )
            else:
                norm = self.k1
            total = 0.0
            for term in terms:
                freq = freqs.get(term, 0)
                if freq:
                    total += (
                        self._idf.get(term, 0.0) * freq * (self.k1 + 1) / (freq + norm)
                    )
            scores[i] = total
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [_hit(self.chunks[int(i)], float(scores[i])) for i in order]


def _min_max(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    # All-equal includes the case that matters: BM25 matching nothing at all.
    # Flattening to zero keeps that from contributing to the blend.
    if high == low:
        return [0.0] * len(scores)
    return [(s - low) / (high - low) for s in scores]


def fuse(
    semantic: list[SearchHit], lexical: list[SearchHit], alpha: float = 0.65
) -> list[SearchHit]:
    """Blend the two rankings: alpha 1.0 is pure semantic, 0.0 pure BM25.

    Each side is min-max normalised first, because BM25 scores are unbounded
    while cosine sits in [-1, 1] -- unnormalised, the lexical side would win on
    magnitude alone regardless of alpha.
    """
    sem = {h.chunk_id: h for h in semantic}
    lex = {h.chunk_id: h for h in lexical}
    sem_scores = dict(zip(sem, _min_max([h.score for h in sem.values()])))
    lex_scores = dict(zip(lex, _min_max([h.score for h in lex.values()])))

    blended = [
        replace(
            sem.get(chunk_id) or lex[chunk_id],
            score=alpha * sem_scores.get(chunk_id, 0.0)
            + (1 - alpha) * lex_scores.get(chunk_id, 0.0),
        )
        for chunk_id in {**sem, **lex}
    ]
    # chunk_id breaks ties so the order is reproducible.
    return sorted(blended, key=lambda h: (-h.score, h.chunk_id))
