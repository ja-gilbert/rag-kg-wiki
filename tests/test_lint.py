from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from core.config import load_config
from core.corpus import Document, load_corpus
from kgraph.extract import Triple, extract_triples
from kgraph.ontology import Ontology
from wikigen import compile as compile_module
from wikigen.compile import compile_wiki
from wikigen.library import IndexEntry, WikiLibrary
from wikigen.lint import CHECKS, LintReport, lint, lint_wiki
from wikigen.page import WikiPage

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"
TODAY = dt.date(2026, 1, 1)


# --------------------------------------------------------------------------
# the real compiled wiki
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    built = load_config(CONFIG)
    built["paths"]["wiki"] = str(tmp_path_factory.mktemp("lint_root") / "wiki")
    compile_wiki(built)
    return built


@pytest.fixture(scope="module")
def report(cfg) -> LintReport:
    return lint_wiki(cfg)


@pytest.fixture(scope="module")
def ontology(cfg) -> Ontology:
    return Ontology.load(cfg["paths"]["ontology"])


@pytest.fixture(scope="module")
def documents(cfg) -> list[Document]:
    return load_corpus(Path(cfg["paths"]["raw"]))


def test_lint_runs_every_check_the_architecture_documents():
    assert set(CHECKS) == {
        "orphan_pages",
        "broken_links",
        "contradictions",
        "stale_pages",
        "superseded_refs",
        "coverage_gaps",
        "index_drift",
    }


def test_the_compiled_wiki_has_no_errors(report):
    assert report.errors == (), report.render()


def test_the_compiled_wiki_has_no_broken_links_orphans_or_gaps(report):
    assert report.of("broken_links") == ()
    assert report.of("orphan_pages") == ()
    assert report.of("coverage_gaps") == ()
    assert report.of("index_drift") == ()


def test_the_planted_superseded_policy_is_acknowledged_wherever_it_is_cited(report):
    # pol-retention-v2 opens "Superseded. Retained for historical reference
    # only." and v3 says it supersedes v2 -- both in prose. Every page that
    # links to v2 has to pass that on, or lint says so.
    assert report.of("superseded_refs") == ()


def test_lint_records_itself_in_the_append_only_log(cfg, report):
    assert "LINT" in (Path(cfg["paths"]["wiki"]) / "log.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# the acceptance case: a contradiction injected as a triple
# --------------------------------------------------------------------------


def test_injecting_a_fake_owned_by_triple_makes_the_contradiction_check_fire(
    monkeypatch, tmp_path
):
    # Step 8's acceptance criterion. Injected at the extraction layer rather
    # than written straight into a page, so this also pins that a bad triple
    # actually reaches the wiki -- triple, to page, to finding.
    fake = Triple(
        subject="Atlas",
        predicate="owned_by",
        object="Data Team",
        doc_id="svc-atlas",
        sentence="Atlas is owned by the Data Team.",
        pattern="{s} is owned by {o}",
    )
    monkeypatch.setattr(
        compile_module,
        "extract_triples",
        lambda documents, ontology: extract_triples(documents, ontology) + [fake],
    )

    built = load_config(CONFIG)
    built["paths"]["wiki"] = str(tmp_path / "wiki")
    compile_wiki(built)

    findings = lint_wiki(built).of("contradictions")
    assert [f.subject for f in findings] == ["Atlas"]
    assert "Data Team" in findings[0].detail
    assert "Platform Team" in findings[0].detail


def test_a_contradiction_is_an_error_so_lint_fails_ci(ontology):
    report = _lint(_atlas_with_two_owners(), ontology)
    assert not report.ok
    assert report.errors and all(f.severity == "error" for f in report.errors)


def test_an_inverse_label_with_several_objects_is_not_a_contradiction(ontology):
    # One team owns three services. `owns` is the inverse of a functional
    # predicate, not a functional predicate -- only the forward direction is
    # constrained, and flagging both would bury every real contradiction.
    team = _page(
        "Platform Team",
        sections=(("Relationships", "- owns [[Atlas]]\n- owns [[Ember]]\n- owns [[Beacon]]"),),
    )
    assert _lint(_library(team, _page("Atlas"), _page("Ember"), _page("Beacon")), ontology).of(
        "contradictions"
    ) == ()


# --------------------------------------------------------------------------
# the other checks, on wikis built to break exactly one of them
# --------------------------------------------------------------------------


def test_a_link_to_a_page_that_does_not_exist_is_reported(ontology):
    atlas = _page("Atlas", sections=(("Related", "[[Ember]]"),))
    findings = _lint(_library(atlas), ontology).of("broken_links")
    assert [(f.subject, f.severity) for f in findings] == [("Atlas", "error")]
    assert "Ember" in findings[0].detail


def test_an_entity_page_nothing_links_to_is_an_orphan(ontology):
    atlas = _page("Atlas", sections=(("Related", "[[Ember]]"),))
    findings = _lint(_library(atlas, _page("Ember")), ontology).of("orphan_pages")
    assert [f.subject for f in findings] == ["Atlas"]
    assert all(f.severity == "warning" for f in findings)


def test_a_topic_page_is_not_an_orphan_for_being_an_entry_point(ontology):
    # Topic pages are reached from the index and link outward; nothing links
    # back to them by design. All four would otherwise be permanent warnings.
    overview = _page("Architecture Overview", page_type="topic", sections=(("Related", "[[Atlas]]"),))
    atlas = _page("Atlas", sections=(("Related", "[[Architecture Overview]]"),))
    assert _lint(_library(overview, atlas), ontology).of("orphan_pages") == ()


def test_a_page_whose_sources_lag_the_corpus_is_stale(ontology):
    documents = [
        _document("svc-atlas", dt.date(2025, 1, 1)),
        _document("svc-ember", dt.date(2026, 1, 1)),
    ]
    findings = _lint(
        _library(_page("Atlas", sources=("svc-atlas",)), _page("Ember", sources=("svc-ember",))),
        ontology,
        documents=documents,
        stale_after_days=180,
    ).of("stale_pages")
    assert [(f.subject, f.severity) for f in findings] == [("Atlas", "warning")]


def test_staleness_is_measured_against_the_corpus_not_the_wall_clock(ontology):
    # Every document in this corpus predates today by more than the configured
    # window, so measuring against `date.today()` would mark all 36 pages stale
    # and re-mark a different set every day the demo is left alone.
    documents = [_document("svc-atlas", dt.date(2020, 1, 1))]
    assert _lint(
        _library(_page("Atlas", sources=("svc-atlas",))),
        ontology,
        documents=documents,
        stale_after_days=1,
    ).of("stale_pages") == ()


def test_citing_a_superseded_page_without_saying_so_is_an_error(ontology):
    v3 = _page("Data Retention Policy v3", sections=(("Relationships", "- supersedes [[Data Retention Policy v2]]"),))
    v2 = _page("Data Retention Policy v2")
    beacon = _page("Beacon", sections=(("History", "Retention follows [[Data Retention Policy v2]]."),))
    findings = _lint(_library(v3, v2, beacon), ontology).of("superseded_refs")
    assert [(f.subject, f.severity) for f in findings] == [("Beacon", "error")]


def test_citing_a_superseded_page_and_saying_so_is_fine(ontology):
    v3 = _page("Data Retention Policy v3", sections=(("Relationships", "- supersedes [[Data Retention Policy v2]]"),))
    v2 = _page("Data Retention Policy v2")
    beacon = _page(
        "Beacon",
        sections=(("History", "Retention followed [[Data Retention Policy v2]], superseded by [[Data Retention Policy v3]]."),),
    )
    assert _lint(_library(v3, v2, beacon), ontology).of("superseded_refs") == ()


def test_a_document_no_page_cites_is_a_coverage_gap(ontology):
    documents = [_document("svc-atlas"), _document("svc-ember")]
    findings = _lint(
        _library(_page("Atlas", sources=("svc-atlas",))), ontology, documents=documents
    ).of("coverage_gaps")
    assert [(f.subject, f.severity) for f in findings] == [("svc-ember", "warning")]


def test_a_page_missing_from_the_index_is_drift(ontology):
    library = WikiLibrary(entries=(), pages={"Atlas": _page("Atlas")})
    findings = _lint(library, ontology).of("index_drift")
    assert [f.subject for f in findings] == ["Atlas"]


def test_an_index_entry_with_no_page_is_drift(ontology):
    library = WikiLibrary(entries=(IndexEntry("Ember", "Ember is a store.", "Service"),), pages={})
    findings = _lint(library, ontology).of("index_drift")
    assert [f.subject for f in findings] == ["Ember"]


def test_an_index_summary_the_page_no_longer_carries_is_drift(ontology):
    atlas = _page("Atlas", summary="Atlas is the ledger service.")
    library = WikiLibrary(
        entries=(IndexEntry("Atlas", "Atlas was the ledger service.", "Service"),),
        pages={"Atlas": atlas},
    )
    assert [f.subject for f in _lint(library, ontology).of("index_drift")] == ["Atlas"]


# --------------------------------------------------------------------------
# the CI contract
# --------------------------------------------------------------------------


def test_warnings_alone_do_not_fail_the_run(ontology):
    documents = [_document("svc-atlas"), _document("svc-ember")]
    report = _lint(_library(_page("Atlas", sources=("svc-atlas",))), ontology, documents=documents)
    assert report.warnings
    assert report.ok


def test_the_command_line_exits_non_zero_on_an_error(monkeypatch, tmp_path, cfg):
    from scripts import lint as lint_script

    monkeypatch.setattr(lint_script, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        lint_script,
        "lint_wiki",
        lambda config: _lint(_atlas_with_two_owners(), Ontology.load(cfg["paths"]["ontology"])),
    )
    with pytest.raises(SystemExit) as caught:
        lint_script.main([])
    assert caught.value.code == 1


def test_the_command_line_exits_zero_on_the_compiled_wiki(cfg, report):
    from scripts import lint as lint_script

    assert lint_script.exit_code(report) == 0


def test_the_report_names_the_check_and_the_page(ontology):
    rendered = _lint(_atlas_with_two_owners(), ontology).render()
    assert "contradictions" in rendered
    assert "Atlas" in rendered


# --------------------------------------------------------------------------
# helpers: small wikis built by hand, one broken thing each
# --------------------------------------------------------------------------


def _page(
    title: str,
    *,
    page_type: str = "entity",
    summary: str = "",
    sources: tuple[str, ...] = ("svc-atlas",),
    sections: tuple[tuple[str, str], ...] = (),
) -> WikiPage:
    return WikiPage(
        title=title,
        page_type=page_type,
        entity_type="Service" if page_type == "entity" else None,
        sources=sources,
        updated=TODAY,
        summary=summary or f"{title} is a thing.",
        sections=sections,
    )


def _library(*pages: WikiPage) -> WikiLibrary:
    return WikiLibrary(
        entries=tuple(IndexEntry(p.title, p.summary, p.page_type) for p in pages),
        pages={p.title: p for p in pages},
    )


def _document(doc_id: str, date: dt.date = TODAY) -> Document:
    return Document(
        doc_id=doc_id,
        title=doc_id,
        doc_type="service",
        date=date,
        body="prose",
        path=Path(f"{doc_id}.txt"),
    )


def _atlas_with_two_owners() -> WikiLibrary:
    atlas = _page(
        "Atlas",
        sections=(("Relationships", "- owned by [[Platform Team]]\n- owned by [[Data Team]]"),),
    )
    return _library(atlas, _page("Platform Team"), _page("Data Team"))


def _lint(
    library: WikiLibrary,
    ontology: Ontology,
    documents: list[Document] | None = None,
    stale_after_days: int = 180,
) -> LintReport:
    cited = sorted({s for title in library.titles for s in library.page(title).sources})
    return lint(
        library,
        documents if documents is not None else [_document(doc_id) for doc_id in cited],
        ontology,
        stale_after_days=stale_after_days,
    )
