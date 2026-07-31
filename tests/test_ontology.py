from __future__ import annotations

from pathlib import Path

import pytest

from kgraph.ontology import Gazetteer, Ontology

ONTOLOGY = Path(__file__).resolve().parent.parent / "data" / "ontology.yaml"


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    return Ontology.load(ONTOLOGY)


def test_load_reads_every_declared_entity(ontology):
    atlas = ontology.entity("Atlas")
    assert atlas.type == "Service"
    assert "the gateway" in atlas.aliases


def test_load_reads_relation_patterns_and_labels(ontology):
    owned_by = ontology.relations["owned_by"]
    assert owned_by.label == "owned by"
    assert owned_by.inverse_label == "owns"
    assert "{s} is owned by {o}" in [p.template for p in owned_by.patterns]


def test_the_inverse_marker_is_stripped_from_the_template(ontology):
    # "{s} owns {o}||inverse" means: match that surface form, then emit the
    # triple the other way round. The marker is syntax, not part of the regex.
    owns = next(
        p for p in ontology.relations["owned_by"].patterns if "owns" in p.template
    )
    assert owns.inverse is True
    assert "||" not in owns.template


def test_gazetteer_finds_entities_by_any_alias(ontology):
    found = Gazetteer(ontology).find("Atlas is owned by the Platform Team.")
    assert [m.name for m in found] == ["Atlas", "Platform Team"]


def test_longest_alias_wins_when_two_overlap(ontology):
    # "Dr. Elena Vasquez" contains "Elena Vasquez" which contains "Elena".
    # Matching the shortest would leave the rest of the name dangling as text.
    found = Gazetteer(ontology).find("Dr. Elena Vasquez is the chief technology officer.")
    assert [(m.name, m.surface) for m in found] == [
        ("Elena Vasquez", "Dr. Elena Vasquez")
    ]


def test_gazetteer_does_not_match_inside_a_longer_word(ontology):
    # "v2"/"v3" are policy aliases and are short enough to appear as substrings.
    assert Gazetteer(ontology).find("The rev3 branch is stale.") == []


def test_every_relation_declares_a_domain_and_a_range(ontology):
    # Type constraints are what stop a mis-signed inverse pattern from asserting
    # "Platform Team owned_by Atlas", so a relation without them is a hole.
    for predicate, relation in ontology.relations.items():
        assert relation.domain, f"{predicate} declares no domain"
        assert relation.range, f"{predicate} declares no range"
