from __future__ import annotations

from pathlib import Path

import pytest

from core.chunking import chunk_documents
from core.corpus import Document, load_corpus

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def _doc(body: str, doc_id: str = "svc-fake") -> Document:
    return Document(
        doc_id=doc_id,
        title="Fake Service",
        doc_type="service",
        date=None,
        body=body,
        path=Path(f"{doc_id}.txt"),
    )


def test_paragraph_strategy_emits_one_chunk_per_paragraph():
    # Paragraph chunks keep their internal hard-wrap newlines; only the blank
    # line between paragraphs is a boundary.
    chunks = chunk_documents(
        [_doc("First para line one.\nline two.\n\nSecond para.\n\n\nThird para.")],
        {"strategy": "paragraph"},
    )
    assert [c.text for c in chunks] == [
        "First para line one.\nline two.",
        "Second para.",
        "Third para.",
    ]


def test_sentence_window_respects_window_and_stride():
    # Five sentences, window 3, stride 2: the second window already reaches the
    # last sentence, so no redundant tail chunk is emitted.
    chunks = chunk_documents(
        [_doc("One. Two. Three. Four. Five.")],
        {"strategy": "sentence_window", "window": 3, "stride": 2},
    )
    assert [c.text for c in chunks] == [
        "One. Two. Three.",
        "Three. Four. Five.",
    ]


def test_fixed_strategy_uses_character_windows_with_overlap():
    # Step is size - overlap = 3, and the third window reaches the end.
    chunks = chunk_documents(
        [_doc("abcdefghij")],
        {"strategy": "fixed", "size": 4, "overlap": 1},
    )
    assert [c.text for c in chunks] == ["abcd", "defg", "ghij"]


def test_fixed_strategy_splits_mid_word_by_design():
    # The whole reason `fixed` is in the config: it cuts wherever the character
    # window lands, which is what makes retrieval quality visibly drop.
    chunks = chunk_documents(
        [_doc("Atlas depends on Ember for authentication.")],
        {"strategy": "fixed", "size": 20, "overlap": 0},
    )
    assert chunks[0].text == "Atlas depends on Emb"


def test_fixed_strategy_rejects_an_overlap_that_cannot_advance():
    # overlap >= size means the window never moves. Python's own message for
    # this ("range() arg 3 must not be zero") tells the reader nothing about
    # which config key is wrong.
    with pytest.raises(ValueError) as excinfo:
        chunk_documents(
            [_doc("abcdefghij")],
            {"strategy": "fixed", "size": 4, "overlap": 4},
        )
    message = str(excinfo.value)
    assert "overlap" in message and "size" in message


def test_unknown_strategy_raises_naming_the_strategy():
    with pytest.raises(ValueError) as excinfo:
        chunk_documents([_doc("Some prose.")], {"strategy": "semantic"})
    assert "semantic" in str(excinfo.value)


# The step's acceptance criteria, run against every strategy at the values
# config.yaml actually ships.
_STRATEGIES = [
    {"strategy": "paragraph"},
    {"strategy": "sentence_window", "window": 3, "stride": 2},
    {"strategy": "fixed", "size": 900, "overlap": 150},
]
_IDS = [c["strategy"] for c in _STRATEGIES]


@pytest.mark.parametrize("cfg", _STRATEGIES, ids=_IDS)
def test_every_strategy_chunks_every_document_in_the_corpus(cfg):
    corpus = load_corpus(RAW)
    chunks = chunk_documents(corpus, cfg)
    assert chunks
    assert {c.doc_id for c in chunks} == {d.doc_id for d in corpus}


@pytest.mark.parametrize("cfg", _STRATEGIES, ids=_IDS)
def test_chunk_ids_are_unique_across_the_corpus(cfg):
    ids = [c.chunk_id for c in chunk_documents(load_corpus(RAW), cfg)]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("cfg", _STRATEGIES, ids=_IDS)
def test_every_chunk_traces_back_to_its_source_document(cfg):
    corpus = load_corpus(RAW)
    titles = {d.doc_id: d.title for d in corpus}
    chunks = chunk_documents(corpus, cfg)
    for c in chunks:
        assert c.doc_id in titles, f"{c.chunk_id} cites an unknown doc_id"
        assert c.doc_title == titles[c.doc_id]
        assert c.text.strip(), f"{c.chunk_id} is blank"
