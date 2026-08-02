from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from approaches.rag import RagApproach
from core.chunking import Chunk, chunk_documents
from core.config import load_config
from core.corpus import load_corpus
from core.embeddings import build_embedder
from core.llm import ExtractiveBackend, Evidence, Generation
from core.vectorstore import BM25Index, VectorStore

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"
RAW = ROOT / "data" / "raw"

_CHUNKS = [
    Chunk("bug-903#0", "bug-903", "BUG-903", "Marcus Chen fixed BUG-903 in Ember."),
    Chunk("svc-atlas#0", "svc-atlas", "Atlas", "Atlas depends on Ember for search."),
    Chunk("pol-ret#0", "pol-ret", "Retention", "The retention policy covers log storage."),
]

# Unit vectors, one per chunk, in a space wide enough that a query can sit far
# from all three -- which is what a question the corpus cannot answer looks like.
_VECTORS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
    ]
)

_STRONG = "who fixed BUG-903?"
_WEAK = "what was the revenue last quarter?"

_QUERY_VECTORS = {
    # cosine 0.9 against the BUG-903 chunk
    _STRONG: [0.9, 0.4359, 0.0, 0.0, 0.0],
    # cosine 0.2 against its own best match: a shrug, but a confident-looking one
    _WEAK: [0.2, 0.0, 0.0, 0.9, 0.3873],
    "log storage retention policy": [0.2, 0.0, 0.0, 0.9, 0.3873],
}


class _FixedEmbedder:
    """Encodes only the questions a test asks about.

    A stand-in rather than a mock: `Embedder` is a one-method protocol, and
    fixing the vectors is what lets a test state an exact cosine score and
    then assert on the confidence rule that reads it.
    """

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([_QUERY_VECTORS[t] for t in texts], dtype=float)


def _rag(llm=None, **overrides) -> RagApproach:
    cfg = {"top_k": 2, "mode": "semantic", "alpha": 0.65, "low_confidence_threshold": 0.25}
    return RagApproach(
        store=VectorStore(_CHUNKS, _VECTORS),
        bm25=BM25Index(_CHUNKS),
        embedder=_FixedEmbedder(),
        llm=llm or ExtractiveBackend(),
        cfg={**cfg, **overrides},
    )


# --------------------------------------------------------------------------
# the envelope
# --------------------------------------------------------------------------


def test_the_answer_identifies_itself_as_rag():
    answer = _rag().answer(_STRONG)
    assert answer.approach == "rag"
    assert answer.label


def test_the_trace_says_what_it_did_in_plain_english():
    trace = _rag().answer(_STRONG).trace
    assert trace
    assert all(isinstance(step, str) and step.strip() for step in trace)


def test_the_timing_covers_the_whole_query():
    ms = _rag().answer(_STRONG).ms
    assert ms["total"] >= 0
    assert {"embed", "retrieve", "generate"} <= set(ms)


def test_the_token_estimate_reflects_the_evidence_handed_over():
    answer = _rag().answer(_STRONG)
    assert answer.tokens_est == sum(len(e["text"]) for e in answer.evidence) // 4


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


def test_both_retrievers_run_even_though_one_mode_answers():
    detail = _rag(mode="semantic").answer(_STRONG).detail
    assert detail["semantic"]
    assert detail["lexical"]


def test_the_configured_mode_is_the_one_that_answers():
    answer = _rag(mode="semantic").answer(_STRONG)
    assert answer.detail["mode"] == "semantic"
    assert answer.evidence[0]["chunk_id"] == "bug-903#0"


def test_lexical_mode_answers_from_bm25():
    answer = _rag(mode="lexical").answer("log storage retention policy")
    assert answer.detail["mode"] == "lexical"
    assert answer.evidence[0]["doc_id"] == "pol-ret"


def test_only_hybrid_mode_fuses_the_two_rankings():
    assert _rag(mode="semantic").answer(_STRONG).detail["fused"] is None
    assert _rag(mode="hybrid").answer(_STRONG).detail["fused"]


def test_top_k_bounds_the_evidence():
    assert len(_rag(top_k=1).answer(_STRONG).evidence) == 1


def test_evidence_carries_its_chunk_document_and_score():
    first = _rag().answer(_STRONG).evidence[0]
    assert {"chunk_id", "doc_id", "doc_title", "text", "score"} <= set(first)
    assert first["score"] == pytest.approx(0.9, abs=1e-3)


def test_citations_are_the_source_documents_without_repeats():
    citations = _rag().answer(_STRONG).citations
    assert citations == list(dict.fromkeys(citations))
    assert "bug-903" in citations


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown rag mode"):
        _rag(mode="telepathy").answer(_STRONG)


# --------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------


def test_a_strong_retrieval_is_confident_and_needs_no_caveat():
    answer = _rag().answer(_STRONG)
    assert answer.confident is True
    assert answer.note is None


def test_a_weak_retrieval_is_not_confident():
    assert _rag().answer(_WEAK).confident is False


def test_the_caveat_quotes_the_score_that_triggered_it():
    # RAG's most consequential failure mode is that vector search always
    # returns something. The number is what makes that visible.
    note = _rag().answer(_WEAK).note
    assert "0.2" in note
    assert "0.25" in note


def test_confidence_reads_cosine_even_when_lexical_answered():
    # BM25 scores are unbounded, so they cannot be compared to a 0.25
    # threshold; the semantic score is the one comparable number.
    answer = _rag(mode="lexical").answer("log storage retention policy")
    assert answer.evidence[0]["doc_id"] == "pol-ret"  # lexical found it fine
    assert answer.confident is False  # but the question is still a semantic shrug


def test_the_top_score_is_reported_whether_or_not_it_is_confident():
    assert _rag().answer(_WEAK).detail["top_score"] == pytest.approx(0.2, abs=1e-3)
    assert _rag().answer(_STRONG).detail["top_score"] == pytest.approx(0.9, abs=1e-3)


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


class _FallingBackBackend:
    """A backend that reports having degraded, the way a dead ollama would."""

    name = "extractive"

    def generate(self, question: str, evidence: list[Evidence]) -> Generation:
        return Generation(
            text="something",
            backend="extractive",
            fell_back=True,
            note="The ollama backend was unavailable.",
        )


def test_a_generator_fallback_is_surfaced_in_the_note():
    # Otherwise the column silently shows extractive output while claiming to
    # be running the configured model.
    answer = _rag(llm=_FallingBackBackend()).answer(_STRONG)
    assert "ollama" in answer.note


def test_a_generator_fallback_does_not_by_itself_destroy_confidence():
    # Confidence is a claim about the retrieval, not about the generator.
    assert _rag(llm=_FallingBackBackend()).answer(_STRONG).confident is True


# --------------------------------------------------------------------------
# against the real corpus
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_rag():
    cfg = load_config(CONFIG)
    chunks = chunk_documents(load_corpus(RAW), cfg["chunking"])
    embedder = build_embedder(cfg["embedding"], [c.text for c in chunks])
    store = VectorStore(chunks, embedder.encode([c.text for c in chunks]))
    return RagApproach(
        store=store,
        bm25=BM25Index(chunks),
        embedder=embedder,
        llm=ExtractiveBackend(max_sentences=cfg["llm"]["max_sentences"]),
        cfg=cfg["rag"],
    )


def test_the_revenue_question_gets_a_confident_looking_answer(corpus_rag):
    # Non-negotiable #7, pinned to what the model actually does. Nothing in the
    # corpus answers this, yet MiniLM puts the question at cosine ~0.47 -- above
    # the vendor and login questions, which are real. So no fixed threshold can
    # flag it without flagging those too, and RAG confidently cites a page about
    # someone who tracks business signals. Saying "no path" is the graph's job;
    # see tests/test_kg.py. This test exists to keep that contrast honest.
    answer = corpus_rag.answer("What was Meridian Systems' revenue last quarter?")
    assert answer.evidence, "vector search always returns something -- that is the point"
    assert answer.confident is True
    assert answer.detail["top_score"] > 0.4


def test_the_similarity_score_is_visible_whatever_the_verdict(corpus_rag):
    # The number is the only thing standing between a real hit and a shrug.
    answer = corpus_rag.answer("What was Meridian Systems' revenue last quarter?")
    assert answer.detail["top_score"] == pytest.approx(answer.evidence[0]["score"], abs=1e-3)


def test_the_sluggish_question_reaches_the_slow_response_runbook(corpus_rag):
    answer = corpus_rag.answer(
        "A customer says the product feels sluggish and unresponsive. What should I check?"
    )
    assert answer.confident is True
    assert any("run-" in c for c in answer.citations)
