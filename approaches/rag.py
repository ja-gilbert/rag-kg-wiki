from __future__ import annotations

from typing import Any

from approaches.base import Answer, Approach, PhaseTimer, estimate_tokens
from core.embeddings import Embedder
from core.llm import Evidence
from core.vectorstore import BM25Index, SearchHit, VectorStore, fuse


class RagApproach(Approach):
    """Embed the question, take the nearest chunks, generate.

    Both retrievers run on every question even though only one mode answers,
    so the UI can put lexical and semantic rankings side by side on the same
    query. At this corpus size the second retrieval is free, and the pair is
    what makes the paraphrase failure visible: "pages take forever to appear"
    shares no content word with the runbook that answers it.
    """

    name = "rag"
    label = "RAG"

    def __init__(
        self,
        store: VectorStore,
        bm25: BM25Index,
        embedder: Embedder,
        llm: Any,
        cfg: dict[str, Any],
    ) -> None:
        self._store = store
        self._bm25 = bm25
        self._embedder = embedder
        self._llm = llm
        self._top_k = cfg.get("top_k", 5)
        self._mode = cfg.get("mode", "semantic")
        self._alpha = cfg.get("alpha", 0.65)
        self._threshold = cfg.get("low_confidence_threshold", 0.25)

    def answer(self, question: str) -> Answer:
        if self._mode not in ("semantic", "lexical", "hybrid"):
            raise ValueError(f"unknown rag mode {self._mode!r}")

        timer = PhaseTimer()
        with timer.phase("embed"):
            query_vector = self._embedder.encode([question])[0]

        with timer.phase("retrieve"):
            semantic = self._store.search(query_vector, self._top_k)
            lexical = self._bm25.search(question, self._top_k)
            fused = fuse(semantic, lexical, self._alpha) if self._mode == "hybrid" else None
            chosen = {"semantic": semantic, "lexical": lexical, "hybrid": fused}[self._mode]

        # The confidence rule always reads the best cosine, whichever mode
        # answered. BM25 scores are unbounded, so there is no threshold that
        # means the same thing for both -- and min-max normalising the lexical
        # side would put its top hit at 1.0 by construction, which would make
        # a garbage retrieval look certain.
        top_score = semantic[0].score if semantic else 0.0
        confident = top_score >= self._threshold

        with timer.phase("generate"):
            generation = self._llm.generate(question, _evidence(chosen))

        return Answer(
            approach=self.name,
            label=self.label,
            answer=generation.text,
            evidence=[_as_dict(hit) for hit in chosen],
            citations=list(dict.fromkeys(hit.doc_id for hit in chosen)),
            trace=self._trace(chosen, top_score, confident),
            detail={
                "mode": self._mode,
                "top_k": self._top_k,
                "semantic": [_as_dict(h) for h in semantic],
                "lexical": [_as_dict(h) for h in lexical],
                "fused": [_as_dict(h) for h in fused] if fused else None,
                "top_score": round(top_score, 4),
                "threshold": self._threshold,
            },
            ms=timer.ms(),
            tokens_est=estimate_tokens(hit.text for hit in chosen),
            confident=confident,
            note=self._note(top_score, confident, generation.note),
        )

    def _trace(self, chosen: list[SearchHit], top_score: float, confident: bool) -> list[str]:
        steps = [
            "Embedded the question into the same vector space as the chunks.",
            f"Scored every chunk by cosine similarity and kept the top {self._top_k}.",
            "Ran BM25 over the same query so the two rankings can be compared.",
        ]
        if self._mode == "hybrid":
            steps.append(f"Fused the two rankings at alpha={self._alpha}.")
        steps.append(
            f"Answered from the {self._mode} ranking: "
            f"{len(chosen)} chunk(s) from {len({h.doc_id for h in chosen})} document(s)."
        )
        if not confident:
            steps.append(
                f"Best cosine was {top_score:.2f}, under the {self._threshold} "
                "threshold, so the answer is flagged as unreliable."
            )
        return steps

    def _note(self, top_score: float, confident: bool, generation_note: str | None) -> str | None:
        notes = []
        if not confident:
            notes.append(
                f"The best chunk scored {top_score:.2f}, below the "
                f"{self._threshold} threshold. Vector search always returns "
                "something, so these chunks may have nothing to do with the "
                "question."
            )
        if generation_note:
            notes.append(generation_note)
        return " ".join(notes) or None


def _evidence(hits: list[SearchHit]) -> list[Evidence]:
    return [Evidence(citation=hit.doc_id, text=hit.text) for hit in hits]


def _as_dict(hit: SearchHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "doc_title": hit.doc_title,
        "text": hit.text,
        "score": round(hit.score, 4),
    }
