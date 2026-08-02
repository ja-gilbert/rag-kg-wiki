from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import load_config
from scripts.artefacts import ArtefactMissing, load_kg, load_rag
from scripts.ask import ask, ask_all
from scripts.build_all import build_all, summary_table
from scripts.build_graph import build_graph
from scripts.build_index import build_index

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"


@pytest.fixture
def cfg(tmp_path):
    built = load_config(CONFIG)
    built["paths"]["build"] = str(tmp_path / "build")
    # The hash backend carries no semantics, which is fine here: these tests are
    # about artefacts existing and loading, not about retrieval quality, and it
    # keeps the suite off the 80MB model download path.
    built["embedding"] = {"backend": "hash", "dim": 64}
    return built


@pytest.fixture
def built(cfg):
    build_all(cfg)
    return cfg


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


def test_the_index_loads_back_with_a_vector_per_chunk(cfg):
    stats = build_index(cfg)
    store = load_rag(cfg)[0]
    assert len(store.chunks) == stats.count
    assert store.vectors.shape[0] == len(store.chunks)


def test_the_graph_artefact_keeps_the_sentence_behind_every_triple(cfg):
    # Non-negotiable #8. An artefact that dropped provenance would make every
    # edge in the UI unauditable.
    build_graph(cfg)
    rows = json.loads((Path(cfg["paths"]["build"]) / "triples.json").read_text(encoding="utf-8"))
    assert rows
    assert all(row["sentence"].strip() and row["doc_id"] for row in rows)


def test_the_graph_loads_back_as_one_connected_component(cfg):
    build_graph(cfg)
    graph, _ = load_kg(cfg)
    assert graph.component_count() == 1


def test_build_all_produces_every_artefact(cfg):
    build_all(cfg)
    build = Path(cfg["paths"]["build"])
    assert (build / "triples.json").exists()
    assert list(build.glob("index/*"))


def test_build_all_reports_one_row_per_approach(cfg):
    rows = build_all(cfg)
    assert {row.approach for row in rows} == {"rag", "kg"}


def test_the_summary_reports_what_each_build_cost(cfg):
    table = summary_table(build_all(cfg))
    assert "rag" in table and "kg" in table
    # The asymmetry is the finding: it belongs on screen, not in a comment.
    assert "seconds" in table.lower() or "s" in table


# --------------------------------------------------------------------------
# missing artefacts
# --------------------------------------------------------------------------


def test_a_missing_index_names_the_command_that_fixes_it(cfg):
    with pytest.raises(ArtefactMissing, match="scripts.build_all"):
        load_rag(cfg)


def test_a_missing_graph_names_the_command_that_fixes_it(cfg):
    with pytest.raises(ArtefactMissing, match="scripts.build_all"):
        load_kg(cfg)


def test_a_missing_artefact_is_not_a_stack_trace_about_npy(cfg):
    with pytest.raises(ArtefactMissing) as caught:
        load_rag(cfg)
    assert ".npy" not in str(caught.value)


# --------------------------------------------------------------------------
# asking
# --------------------------------------------------------------------------


def test_ask_answers_with_every_approach(built):
    answers = ask("Who fixed the bug in the service that Atlas depends on?", built)
    assert [a.approach for a in answers] == ["rag", "kg"]


def test_every_answer_fills_the_shared_envelope(built):
    for answer in ask("What is the guidance on rotating credentials?", built):
        assert answer.trace
        assert answer.ms["total"] >= 0
        assert answer.confident or answer.note


def test_ask_all_answers_every_demo_question(built):
    results = ask_all(built)
    assert len(results) == 10
    assert all(answers for _, answers in results)


def test_the_kg_still_finds_marcus_chen_through_the_built_artefacts(built):
    kg = [a for a in ask("Who fixed the bug in the service that Atlas depends on?", built)][1]
    assert kg.detail["answer_entity"] == "Marcus Chen"
