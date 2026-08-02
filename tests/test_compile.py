from __future__ import annotations

from pathlib import Path

import pytest

from core.config import load_config
from core.corpus import load_corpus
from wikigen.compile import compile_wiki
from wikigen.page import WikiPage

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"
RAW = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    cfg = load_config(CONFIG)
    cfg["paths"]["wiki"] = str(tmp_path_factory.mktemp("wiki_root") / "wiki")
    result = compile_wiki(cfg)
    return cfg, result


@pytest.fixture(scope="module")
def wiki(compiled) -> Path:
    return Path(compiled[0]["paths"]["wiki"])


@pytest.fixture(scope="module")
def pages(wiki) -> list[WikiPage]:
    return [WikiPage.parse(p.read_text(encoding="utf-8")) for p in sorted((wiki / "pages").glob("*.md"))]


@pytest.fixture(scope="module")
def doc_ids() -> set[str]:
    return {d.doc_id for d in load_corpus(RAW)}


# --------------------------------------------------------------------------
# what the compiler is allowed to write
# --------------------------------------------------------------------------


def test_everything_written_is_markdown(wiki):
    assert [p.suffix for p in wiki.rglob("*") if p.is_file()] == [
        ".md" for p in wiki.rglob("*") if p.is_file()
    ]


def test_nothing_is_written_outside_the_wiki_directory(compiled, wiki):
    # Non-negotiable #3: raw stays .txt and untouched, markdown is output only.
    _, result = compiled
    assert all(Path(p).is_relative_to(wiki) for p in result.written)


def test_the_raw_corpus_is_not_touched(doc_ids):
    assert {p.suffix for p in RAW.glob("*")} == {".txt"}


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_no_page_cites_a_document_that_does_not_exist(pages, doc_ids):
    for page in pages:
        assert set(page.sources) <= doc_ids, page.title


def test_every_page_cites_at_least_one_document(pages):
    assert all(page.sources for page in pages)


def test_no_page_cites_the_same_document_twice(pages):
    for page in pages:
        assert len(set(page.sources)) == len(page.sources), page.title


def test_a_page_cites_what_it_was_built_from_not_every_passing_mention(pages):
    # Provenance that lists every document mentioning Atlas is provenance no
    # one can audit: following such a citation finds nothing the page said.
    atlas = next(p for p in pages if p.title == "Atlas")
    assert "svc-atlas" in atlas.sources
    assert "person-priya-raman" not in atlas.sources


def test_every_raw_document_is_cited_by_some_page(pages, doc_ids):
    # The coverage-gap check from step 8's acceptance list, enforced at compile
    # time rather than discovered later by lint.
    cited = {source for page in pages for source in page.sources}
    assert doc_ids - cited == set()


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------


def test_there_is_a_page_per_declared_entity(pages):
    titles = {p.title for p in pages}
    for expected in ("Atlas", "Ember", "Marcus Chen", "Platform Team", "INC-2041", "BUG-903"):
        assert expected in titles


def test_relationship_bullets_come_from_the_knowledge_graph(pages):
    atlas = next(p for p in pages if p.title == "Atlas")
    assert ("depends on", "Ember") in atlas.relationships


def test_an_entity_page_declares_its_ontology_type(pages):
    assert next(p for p in pages if p.title == "Atlas").entity_type == "Service"


def test_every_page_has_a_summary_for_the_index_to_quote(pages):
    assert all(page.summary.strip() for page in pages)


def test_every_wiki_link_resolves_to_a_page(pages):
    titles = {p.title for p in pages}
    for page in pages:
        assert set(page.links) <= titles, f"{page.title} links to a page that does not exist"


def test_the_topic_pages_exist(pages):
    titles = {p.title for p in pages}
    assert {"Architecture Overview", "Incident History", "Vendor Risk", "Who Owns What"} <= titles


def test_the_architecture_topic_page_synthesises_several_documents(pages):
    # This is the page that beats RAG on "how does it all fit together": its
    # content exists in no single source document.
    overview = next(p for p in pages if p.title == "Architecture Overview")
    assert len(overview.sources) >= 5
    assert len(overview.links) >= 5


def test_a_topic_page_has_no_entity_type(pages):
    assert next(p for p in pages if p.title == "Vendor Risk").entity_type is None


# --------------------------------------------------------------------------
# the catalogue and the log
# --------------------------------------------------------------------------


def test_the_index_lists_every_page(wiki, pages):
    index = (wiki / "index.md").read_text(encoding="utf-8")
    for page in pages:
        assert f"[[{page.title}]]" in index


def test_the_index_quotes_each_summary(wiki, pages):
    index = (wiki / "index.md").read_text(encoding="utf-8")
    atlas = next(p for p in pages if p.title == "Atlas")
    assert atlas.summary in index


def test_the_index_groups_pages_by_category(wiki):
    index = (wiki / "index.md").read_text(encoding="utf-8")
    assert "## Service" in index
    assert "## Topics" in index


def test_the_log_records_the_compile(wiki):
    assert "COMPILE" in (wiki / "log.md").read_text(encoding="utf-8")


def test_the_log_is_append_only(compiled, wiki):
    cfg, _ = compiled
    before = (wiki / "log.md").read_text(encoding="utf-8")
    compile_wiki(cfg)
    after = (wiki / "log.md").read_text(encoding="utf-8")
    assert after.startswith(before)
    assert len(after) > len(before)


def test_recompiling_produces_identical_pages(compiled, wiki):
    cfg, _ = compiled
    before = (wiki / "pages" / "atlas.md").read_text(encoding="utf-8")
    compile_wiki(cfg)
    assert (wiki / "pages" / "atlas.md").read_text(encoding="utf-8") == before
