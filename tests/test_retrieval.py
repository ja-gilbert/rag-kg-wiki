from __future__ import annotations

import re
from pathlib import Path

from core.chunking import chunk_documents
from core.corpus import load_corpus
from core.embeddings import build_embedder
from core.vectorstore import BM25Index, VectorStore

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

_TARGET = "run-slow-responses"
_TOP_K = 5  # config.yaml rag.top_k

# NOT the query docs/IMPLEMENTATION.md names. That one ("a customer says the
# product feels sluggish") shares customer / product / feels / slow / check with
# the runbook, so BM25 also ranks it 2nd and the contrast collapses. The spec's
# premise -- that the runbook uses neither "sluggish" nor "unresponsive" -- is
# false: it contains "unresponsive". See the step report.
_QUERY = "pages take forever to appear"

_NEURAL = {
    "backend": "sentence-transformers",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "cache_dir": "models",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def test_the_query_shares_no_content_words_with_the_runbook():
    # Grounds the test below, and compares whole tokens because that is what
    # BM25 matches on: the runbook says "appears", which is not "appear".
    body = next(d for d in load_corpus(RAW) if d.doc_id == _TARGET).body
    assert _tokens(body).isdisjoint({"pages", "forever", "appear", "laggy"})


def test_semantic_retrieval_finds_what_lexical_search_ranks_out_of_reach():
    """The test that proves the embedding space carries meaning.

    Cosine puts this runbook first. BM25 does find a thread of lexical signal
    (via common words like "to"), but buries it far outside the top-k window
    the RAG approach actually reads, so in practice lexical search misses it.
    """
    corpus = load_corpus(RAW)
    chunks = chunk_documents(corpus, {"strategy": "paragraph"})
    texts = [c.text for c in chunks]

    embedder = build_embedder(_NEURAL, corpus_texts=texts)
    store = VectorStore(chunks, embedder.encode(texts))

    semantic = store.search(embedder.encode([_QUERY])[0], top_k=3)
    assert _TARGET in {h.doc_id for h in semantic}

    lexical = BM25Index(chunks).search(_QUERY, top_k=len(chunks))
    assert _TARGET not in {h.doc_id for h in lexical[:_TOP_K]}

    # And not merely just below the cut: it sits far down the lexical ranking.
    lexical_docs = list(dict.fromkeys(h.doc_id for h in lexical))
    assert lexical_docs.index(_TARGET) > 10
