from __future__ import annotations

import numpy as np
import pytest

from core.chunking import Chunk
from core.vectorstore import BM25Index, SearchHit, VectorStore, fuse


def _chunks() -> list[Chunk]:
    return [
        Chunk("svc-atlas#0", "svc-atlas", "Atlas API Gateway", "Atlas is the edge."),
        Chunk("svc-ember#0", "svc-ember", "Ember Auth", "Ember authenticates."),
        Chunk("svc-beacon#0", "svc-beacon", "Beacon", "Beacon aggregates."),
    ]


def test_search_ranks_by_cosine_similarity():
    # Vectors are L2-normalised by the embedder, so the store's dot product is
    # cosine directly -- no per-query normalisation, one matrix multiply.
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7071068, 0.7071068]], dtype=np.float32
    )
    store = VectorStore(_chunks(), vectors)
    hits = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert [h.chunk_id for h in hits] == ["svc-atlas#0", "svc-beacon#0"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.7071068, abs=1e-6)


def test_store_round_trips_through_save_and_load(tmp_path):
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7071068, 0.7071068]], dtype=np.float32
    )
    store = VectorStore(_chunks(), vectors)
    store.save(tmp_path / "index")

    restored = VectorStore.load(tmp_path / "index")
    assert [c.chunk_id for c in restored.chunks] == [c.chunk_id for c in store.chunks]
    assert [c.doc_title for c in restored.chunks] == [
        c.doc_title for c in store.chunks
    ]
    assert np.allclose(restored.vectors, store.vectors)

    query = np.array([1.0, 0.0], dtype=np.float32)
    assert [h.chunk_id for h in restored.search(query, top_k=3)] == [
        h.chunk_id for h in store.search(query, top_k=3)
    ]


def test_loading_a_missing_index_says_how_to_build_it(tmp_path):
    # A demo that dies on a missing .npy with a bare traceback is a demo nobody
    # gets running.
    with pytest.raises(FileNotFoundError) as excinfo:
        VectorStore.load(tmp_path / "nothing-here")
    assert "build_all" in str(excinfo.value)


def test_mismatched_chunks_and_vectors_are_rejected():
    with pytest.raises(ValueError):
        VectorStore(_chunks(), np.zeros((2, 4), dtype=np.float32))


def test_bm25_ranks_on_literal_term_overlap():
    hits = BM25Index(_chunks()).search("Ember authenticates", top_k=3)
    assert hits[0].chunk_id == "svc-ember#0"
    assert hits[0].score > 0


def test_bm25_scores_zero_when_no_term_matches():
    # The lexical counterpart to RAG's "a similarity score is never zero": BM25
    # genuinely can say it found nothing, and the UI should be able to show that.
    hits = BM25Index(_chunks()).search("quarterly revenue forecast", top_k=3)
    assert all(h.score == 0 for h in hits)


def test_fuse_blends_normalised_scores_by_alpha():
    semantic = [
        SearchHit("a#0", "a", "A", "", 1.0),
        SearchHit("b#0", "b", "B", "", 0.0),
    ]
    lexical = [
        SearchHit("a#0", "a", "A", "", 0.0),
        SearchHit("b#0", "b", "B", "", 4.0),
    ]
    # alpha=1.0 is pure semantic, 0.0 pure lexical, so the winner flips.
    assert fuse(semantic, lexical, alpha=1.0)[0].chunk_id == "a#0"
    assert fuse(semantic, lexical, alpha=0.0)[0].chunk_id == "b#0"

    # Each list is min-max normalised before blending, so BM25's unbounded
    # scores cannot swamp cosine's [-1, 1] simply by being larger numbers.
    even = fuse(semantic, lexical, alpha=0.5)
    assert [h.score for h in even] == [pytest.approx(0.5), pytest.approx(0.5)]
