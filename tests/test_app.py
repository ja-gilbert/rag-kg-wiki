from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app, scoreboard
from core.config import load_config
from scripts.build_all import build_all

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"
QUESTION = "Who fixed the bug in the service that Atlas depends on?"


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    built = load_config(CONFIG)
    root = tmp_path_factory.mktemp("lab")
    built["paths"]["build"] = str(root / "build")
    built["paths"]["wiki"] = str(root / "wiki")
    # No semantics needed here and no 80MB download either: these tests are
    # about the API contract, not about retrieval quality.
    built["embedding"] = {"backend": "hash", "dim": 64}
    build_all(built)
    return built


@pytest.fixture(scope="module")
def client(cfg) -> TestClient:
    return TestClient(create_app(cfg))


@pytest.fixture(scope="module")
def answered(client) -> dict:
    return client.post("/api/ask", json={"question": QUESTION}).json()


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------


def test_asking_returns_all_three_approaches_in_a_fixed_order(answered):
    assert [a["approach"] for a in answered["answers"]] == ["rag", "kg", "wiki"]


def test_every_answer_arrives_in_the_shared_envelope(answered):
    from approaches.base import ENVELOPE_FIELDS

    for answer in answered["answers"]:
        assert set(answer) == set(ENVELOPE_FIELDS)


def test_an_answer_that_is_not_confident_carries_its_note(answered):
    for answer in answered["answers"]:
        assert answer["confident"] or answer["note"]


def test_the_kg_column_still_finds_marcus_chen_over_http(answered):
    kg = next(a for a in answered["answers"] if a["approach"] == "kg")
    assert kg["detail"]["answer_entity"] == "Marcus Chen"


def test_an_empty_question_is_rejected(client):
    assert client.post("/api/ask", json={"question": "   "}).status_code == 422


def test_the_demo_questions_are_served_with_their_annotations(client):
    # `expect` and `why` are what makes the demo teachable rather than a list
    # of strings: the UI can say which approach this one is meant to favour.
    rows = client.get("/api/questions").json()["questions"]
    assert len(rows) == 10
    assert all(row["q"] and row["expect"] and row["why"] for row in rows)


def test_every_demo_question_answers_with_three_populated_columns(client):
    # Step 9's headline acceptance criterion, over the API the page will call.
    for row in client.get("/api/questions").json()["questions"]:
        answers = client.post("/api/ask", json={"question": row["q"]}).json()["answers"]
        assert len(answers) == 3
        assert all(a["trace"] for a in answers)


# --------------------------------------------------------------------------
# the scoreboard
# --------------------------------------------------------------------------


def test_the_scoreboard_has_a_row_per_approach(answered):
    assert [r["approach"] for r in answered["scoreboard"]] == ["rag", "kg", "wiki"]


def test_the_scoreboard_reports_what_each_architecture_cost(answered):
    for row in answered["scoreboard"]:
        assert row["latency_ms"] >= 0
        assert row["evidence_tokens"] >= 0
        assert row["documents_touched"] >= 0
        assert isinstance(row["derivation_shown"], bool)
        assert isinstance(row["admitted_ignorance"], bool)


def test_only_the_graph_shows_a_derivation_for_a_multi_hop_question(answered):
    shown = {r["approach"]: r["derivation_shown"] for r in answered["scoreboard"]}
    assert shown == {"rag": False, "kg": True, "wiki": False}


def test_documents_touched_counts_distinct_sources(answered):
    for answer, row in zip(answered["answers"], answered["scoreboard"]):
        assert row["documents_touched"] == len(set(answer["citations"]))


# --------------------------------------------------------------------------
# one broken approach must not blank the other two
# --------------------------------------------------------------------------


class Broken:
    """An approach that raises whatever it is asked."""

    name, label = "rag", "RAG"

    def answer(self, question: str):
        raise RuntimeError("the index melted")


@pytest.fixture
def with_broken_rag(client, monkeypatch) -> dict:
    lab = client.app.state.lab
    monkeypatch.setattr(
        client.app.state,
        "lab",
        replace(lab, approaches=(Broken(), *lab.approaches[1:])),
    )
    return client.post("/api/ask", json={"question": QUESTION}).json()


def test_a_failing_approach_leaves_the_other_columns_standing(with_broken_rag):
    answers = with_broken_rag["answers"]
    assert [a["approach"] for a in answers] == ["rag", "kg", "wiki"]
    assert "the index melted" in answers[0]["note"]
    assert answers[1]["answer"] and answers[2]["answer"]


def test_a_failed_approach_admits_it_rather_than_looking_confident(with_broken_rag):
    assert with_broken_rag["answers"][0]["confident"] is False
    assert with_broken_rag["scoreboard"][0]["admitted_ignorance"] is True


# --------------------------------------------------------------------------
# the other tabs
# --------------------------------------------------------------------------


def test_the_graph_is_served_with_the_sentence_behind_every_edge(client):
    body = client.get("/api/graph").json()
    assert body["nodes"] and body["edges"]
    # Non-negotiable #8: an edge you cannot audit back to prose is not evidence.
    assert all(edge["sentence"].strip() and edge["doc_id"] for edge in body["edges"])


def test_the_wiki_catalogue_lists_every_page_with_its_summary(client):
    pages = client.get("/api/wiki").json()["pages"]
    assert len(pages) >= 30
    assert all(page["title"] and page["summary"] for page in pages)


def test_a_wiki_page_carries_its_links_backlinks_and_sources(client):
    page = client.get("/api/wiki/Atlas").json()
    assert page["entity_type"] == "Service"
    assert "svc-atlas" in page["sources"]
    assert "Ember" in page["links"]
    assert page["backlinks"]
    assert page["markdown"].startswith("---")


def test_an_unknown_wiki_page_is_a_404(client):
    assert client.get("/api/wiki/Nonexistent").status_code == 404


def test_lint_is_served_as_grouped_findings(client):
    body = client.get("/api/lint").json()
    assert body["ok"] is True
    assert body["page_count"] >= 30
    assert all(f["check"] and f["severity"] in ("error", "warning") for f in body["findings"])


def test_running_lint_from_the_browser_does_not_write_to_the_log(cfg, client):
    log = Path(cfg["paths"]["wiki"]) / "log.md"
    before = log.read_text(encoding="utf-8")
    client.get("/api/lint")
    client.get("/api/lint")
    assert log.read_text(encoding="utf-8") == before


def test_every_raw_document_is_reachable_from_the_sources_tab(client):
    documents = client.get("/api/sources").json()["documents"]
    assert len(documents) == 39
    assert all(d["doc_id"] and d["title"] and d["doc_type"] for d in documents)


def test_a_source_serves_the_raw_prose_a_citation_points_at(client):
    body = client.get("/api/sources/svc-atlas").json()
    assert body["doc_type"] == "service"
    assert "Atlas" in body["text"]


def test_every_citation_from_every_column_resolves_to_a_document(client, answered):
    known = {d["doc_id"] for d in client.get("/api/sources").json()["documents"]}
    for answer in answered["answers"]:
        assert set(answer["citations"]) <= known, answer["approach"]


def test_an_unknown_document_is_a_404(client):
    assert client.get("/api/sources/no-such-doc").status_code == 404


# --------------------------------------------------------------------------
# status, and the missing-artefacts banner
# --------------------------------------------------------------------------


def test_status_reports_the_configuration_that_produced_the_answers(client):
    body = client.get("/api/status").json()
    assert body["ready"] is True
    assert body["problem"] is None
    assert body["documents"] == 39
    assert [a["name"] for a in body["approaches"]] == ["rag", "kg", "wiki"]
    assert body["config"]["llm"] == "extractive"


def test_a_lab_with_no_artefacts_still_serves_the_shell(tmp_path):
    unbuilt = load_config(CONFIG)
    unbuilt["paths"]["build"] = str(tmp_path / "build")
    unbuilt["paths"]["wiki"] = str(tmp_path / "wiki")
    client = TestClient(create_app(unbuilt))

    status = client.get("/api/status").json()
    assert status["ready"] is False
    assert "build_all" in status["problem"]
    assert client.get("/").status_code == 200


def test_an_unbuilt_lab_still_serves_the_corpus_and_the_questions(tmp_path):
    # The sources tab and the question list read `data/`, which is never
    # generated -- there is no reason to hide them behind the banner.
    unbuilt = load_config(CONFIG)
    unbuilt["paths"]["build"] = str(tmp_path / "build")
    unbuilt["paths"]["wiki"] = str(tmp_path / "wiki")
    client = TestClient(create_app(unbuilt))

    assert len(client.get("/api/sources").json()["documents"]) == 39
    assert len(client.get("/api/questions").json()["questions"]) == 10


def test_an_unbuilt_lab_says_what_to_run_instead_of_answering(tmp_path):
    unbuilt = load_config(CONFIG)
    unbuilt["paths"]["build"] = str(tmp_path / "build")
    unbuilt["paths"]["wiki"] = str(tmp_path / "wiki")
    client = TestClient(create_app(unbuilt))

    for path, call in (
        ("/api/ask", lambda: client.post("/api/ask", json={"question": QUESTION})),
        ("/api/graph", lambda: client.get("/api/graph")),
        ("/api/wiki", lambda: client.get("/api/wiki")),
        ("/api/lint", lambda: client.get("/api/lint")),
    ):
        response = call()
        assert response.status_code == 503, path
        assert "build_all" in response.json()["detail"], path


def test_the_root_serves_something_a_browser_can_render(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.text.strip()


# --------------------------------------------------------------------------
# the scoreboard, away from HTTP
# --------------------------------------------------------------------------


def test_the_scoreboard_reads_the_envelope_not_the_approach_name():
    from approaches.base import Answer

    def answer(name: str, **overrides) -> Answer:
        return Answer(
            **{
                "approach": name,
                "label": name.upper(),
                "answer": "an answer",
                "evidence": [],
                "citations": ["svc-atlas", "svc-atlas", "svc-ember"],
                "trace": ["did a thing"],
                "detail": {},
                "ms": {"total": 12.5},
                "tokens_est": 400,
                "confident": True,
                **overrides,
            }
        )

    rows = scoreboard([answer("rag"), answer("kg", detail={"path": {"text": "A -> B"}})])
    assert [r["derivation_shown"] for r in rows] == [False, True]
    assert [r["documents_touched"] for r in rows] == [2, 2]
