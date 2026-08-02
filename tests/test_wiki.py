from __future__ import annotations

from pathlib import Path

import pytest

from approaches.wiki import WikiApproach
from core.config import load_config
from core.llm import ExtractiveBackend
from wikigen.compile import compile_wiki
from wikigen.library import WikiLibrary

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"

_FIT_TOGETHER = (
    "How do Atlas, Ember, Beacon, Cinder and Delta Store fit together, and who owns what?"
)


@pytest.fixture(scope="module")
def library(tmp_path_factory) -> WikiLibrary:
    cfg = load_config(CONFIG)
    cfg["paths"]["wiki"] = str(tmp_path_factory.mktemp("wiki_root") / "wiki")
    compile_wiki(cfg)
    return WikiLibrary.load(cfg["paths"]["wiki"])


def _wiki(library, **overrides) -> WikiApproach:
    cfg = {"max_pages": 6, "backlink_expansion": 1}
    return WikiApproach(library=library, llm=ExtractiveBackend(), cfg={**cfg, **overrides})


# --------------------------------------------------------------------------
# the catalogue
# --------------------------------------------------------------------------


def test_the_library_reads_every_page_from_the_index(library):
    assert len(library.entries) == 36


def test_an_entry_carries_its_category_and_summary(library):
    atlas = next(e for e in library.entries if e.title == "Atlas")
    assert atlas.category == "Service"
    assert atlas.summary.startswith("Atlas is the public edge")


def test_the_library_loads_a_whole_page_by_title(library):
    assert library.page("Atlas").title == "Atlas"


def test_backlinks_find_the_pages_pointing_at_one(library):
    assert "Atlas" in library.backlinks("Ember")


def test_a_missing_wiki_names_the_command_that_builds_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="compile"):
        WikiLibrary.load(tmp_path / "nothing")


# --------------------------------------------------------------------------
# navigation, not similarity
# --------------------------------------------------------------------------


def test_scoring_reads_titles_and_summaries_only(library):
    # Non-negotiable #4. "load-bearing" appears in the body of the Atlas page
    # and nowhere in its title or summary, so a table-of-contents scan must not
    # find it. If this ever passes, the wiki has quietly become RAG.
    assert "load-bearing" in library.page("Atlas").section("What it is")
    assert "load-bearing" not in next(e for e in library.entries if e.title == "Atlas").summary

    matched = _wiki(library).answer("what is the load-bearing component?").detail["matched"]
    assert "Atlas" not in [m["title"] for m in matched]


def test_the_fit_together_question_lands_on_topic_pages(library):
    # The acceptance criterion: answered from one or two topic pages, where RAG
    # needs five fragments. Topic pages win here on their summaries, not on any
    # special treatment in the scorer.
    matched = _wiki(library).answer(_FIT_TOGETHER).detail["matched"]
    assert matched[0]["page_type"] == "topic"
    assert {"Architecture Overview", "Who Owns What"} <= {m["title"] for m in matched}


def test_the_answer_to_fit_together_covers_every_service(library):
    answer = _wiki(library).answer(_FIT_TOGETHER)
    assert len({c for c in answer.citations}) >= 5


def test_links_are_followed_one_hop_out(library):
    with_hop = _wiki(library, backlink_expansion=1).answer(_FIT_TOGETHER)
    without = _wiki(library, backlink_expansion=0).answer(_FIT_TOGETHER)
    assert len(with_hop.detail["followed"]) > 0
    assert without.detail["followed"] == []


def test_max_pages_caps_the_evidence(library):
    assert len(_wiki(library, max_pages=2).answer(_FIT_TOGETHER).evidence) == 2


def test_evidence_is_whole_pages_not_fragments(library):
    # The whole page's prose, not a chunk of it -- and not its frontmatter
    # either, which is filesystem bookkeeping rather than something a reader
    # reads. That is the difference from RAG: the unit is the page.
    page = _wiki(library).answer(_FIT_TOGETHER).evidence[0]
    whole = library.page(page["title"])
    assert page["text"] == whole.body()
    assert whole.summary in page["text"]


# --------------------------------------------------------------------------
# citing twice
# --------------------------------------------------------------------------


def test_every_evidence_page_carries_the_documents_it_was_built_from(library):
    for page in _wiki(library).answer(_FIT_TOGETHER).evidence:
        assert page["doc_ids"]


def test_citations_are_raw_document_ids_not_page_titles(library):
    # The other two approaches cite doc ids, and the sources tab reads them.
    answer = _wiki(library).answer(_FIT_TOGETHER)
    titles = {m["title"] for m in answer.detail["matched"]}
    assert not (set(answer.citations) & titles)


def test_the_pages_used_are_reported_separately_from_the_documents(library):
    # Two-level provenance: answer -> wiki page -> source documents.
    answer = _wiki(library).answer(_FIT_TOGETHER)
    assert answer.detail["matched"]
    assert answer.citations


# --------------------------------------------------------------------------
# the envelope, and admitting ignorance
# --------------------------------------------------------------------------


def test_a_question_the_index_has_no_term_for_finds_nothing(library):
    # A question sharing no term at all with any title or summary is the case
    # the catalogue scan genuinely detects.
    answer = _wiki(library).answer("What is the airspeed velocity of a swallow?")
    assert answer.confident is False
    assert answer.note
    assert answer.evidence == []


def test_the_revenue_question_matches_only_on_the_company_name(library):
    # Measured behaviour, not the behaviour data/questions.yaml predicts. The
    # revenue question shares "Meridian" and "Systems" with a few summaries, so
    # a catalogue scan does return pages -- and no threshold separates it from
    # real questions, since "What is Delta Store" and "within two hops" match
    # exactly the same fraction of their terms. What the wiki can be honest
    # about is *what* it matched on, which is the company name and nothing else.
    answer = _wiki(library).answer("What was Meridian Systems' revenue last quarter?")
    matched_on = {term for m in answer.detail["matched"] for term in m["matched_on"]}
    assert matched_on <= {"meridian", "system"}
    assert not any(term in matched_on for term in ("revenue", "quarter"))


def test_the_answer_identifies_itself_as_wiki(library):
    answer = _wiki(library).answer(_FIT_TOGETHER)
    assert answer.approach == "wiki"
    assert answer.label


def test_the_trace_says_what_it_did_in_plain_english(library):
    trace = _wiki(library).answer(_FIT_TOGETHER).trace
    assert trace and all(isinstance(step, str) and step.strip() for step in trace)


def test_the_timing_covers_scanning_and_reading(library):
    ms = _wiki(library).answer(_FIT_TOGETHER).ms
    assert {"scan", "read", "generate", "total"} <= set(ms)


def test_the_answer_is_prose_and_not_the_page_frontmatter(library):
    # Found by reading real output, not by a unit test: the approach used to
    # hand `page.render()` to the generator, so the frontmatter block competed
    # for selection as a sentence -- and for a question naming Atlas it won,
    # because the block contains "Atlas" in the title and in every source id.
    answer = _wiki(library).answer("Who owns Atlas and what does it depend on?")
    assert not answer.answer.lstrip().startswith("---")
    for chrome in ("title:", "page_type:", "entity_type:", "updated:"):
        assert chrome not in answer.answer


def test_the_evidence_blocks_are_prose_too(library):
    # What the generator saw and what the UI shows have to be the same text,
    # or the evidence panel stops explaining where the answer came from.
    for block in _wiki(library).answer(_FIT_TOGETHER).evidence:
        assert "---" not in block["text"]
        assert "page_type:" not in block["text"]
