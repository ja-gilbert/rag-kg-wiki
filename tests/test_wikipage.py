from __future__ import annotations

import datetime as dt

import pytest

from wikigen.page import WikiPage

_ENTITY = WikiPage(
    title="Atlas",
    page_type="entity",
    entity_type="Service",
    sources=("svc-atlas", "run-atlas", "inc-2041"),
    updated=dt.date(2026, 7, 31),
    summary="The public edge for every Meridian Systems product.",
    sections=(
        ("What it is", "Atlas terminates every external request.\n\nIt fronts the estate."),
        ("Relationships", "- depends on [[Ember]]\n- owned by [[Platform Team]]"),
        ("History", "Atlas was at the centre of [[INC-2041]]."),
        ("Related", "[[Ember]] · [[Platform Team]]"),
    ),
)

_TOPIC = WikiPage(
    title="Architecture Overview",
    page_type="topic",
    entity_type=None,
    sources=("arch-overview", "svc-atlas"),
    updated=dt.date(2026, 7, 31),
    summary="How the five services fit together.",
    sections=(("The shape of it", "[[Atlas]] is the edge."),),
)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_the_title_appears_in_the_frontmatter_and_as_the_heading():
    rendered = _ENTITY.render()
    assert "title: Atlas" in rendered
    assert "\n# Atlas\n" in rendered


def test_the_summary_is_quoted_so_the_index_can_lift_it():
    assert "> The public edge for every Meridian Systems product." in _ENTITY.render()


def test_every_section_is_rendered_as_a_heading():
    rendered = _ENTITY.render()
    for heading, _ in _ENTITY.sections:
        assert f"## {heading}" in rendered


def test_sources_are_listed_under_their_own_heading():
    rendered = _ENTITY.render()
    assert "## Sources" in rendered
    assert "- svc-atlas" in rendered


def test_a_topic_page_declares_no_entity_type():
    assert "entity_type" not in _TOPIC.render()


# --------------------------------------------------------------------------
# round-tripping -- the query layer parses these back
# --------------------------------------------------------------------------


def test_an_entity_page_round_trips_losslessly():
    assert WikiPage.parse(_ENTITY.render()) == _ENTITY


def test_a_topic_page_round_trips_losslessly():
    assert WikiPage.parse(_TOPIC.render()) == _TOPIC


def test_round_tripping_twice_changes_nothing():
    once = _ENTITY.render()
    assert WikiPage.parse(once).render() == once


def test_sections_keep_their_order():
    parsed = WikiPage.parse(_ENTITY.render())
    assert [h for h, _ in parsed.sections] == [h for h, _ in _ENTITY.sections]


def test_a_blank_line_inside_a_section_survives():
    parsed = WikiPage.parse(_ENTITY.render())
    assert parsed.sections[0][1] == _ENTITY.sections[0][1]


def test_a_page_without_frontmatter_is_rejected_by_name():
    with pytest.raises(ValueError, match="frontmatter"):
        WikiPage.parse("# Atlas\n\n> Nothing here.\n")


# --------------------------------------------------------------------------
# what lint and the query layer read off a page
# --------------------------------------------------------------------------


def test_links_finds_every_wiki_link_on_the_page():
    assert _ENTITY.links == ("Ember", "Platform Team", "INC-2041")


def test_relationships_read_their_label_and_target():
    assert _ENTITY.relationships == (("depends on", "Ember"), ("owned by", "Platform Team"))


def test_a_page_with_no_relationships_section_has_none():
    assert _TOPIC.relationships == ()


def test_a_malformed_relationship_bullet_is_ignored_rather_than_guessed():
    # Lint reports these; the parser's job is not to invent an edge from a
    # bullet that does not name one.
    page = WikiPage.parse(
        _ENTITY.render().replace("- owned by [[Platform Team]]", "- something went wrong here")
    )
    assert page.relationships == (("depends on", "Ember"),)
