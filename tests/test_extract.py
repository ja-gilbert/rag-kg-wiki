from __future__ import annotations

from pathlib import Path

import pytest

from core.corpus import Document, load_corpus
from kgraph.extract import extract_triples, extraction_stats
from kgraph.ontology import Ontology

ONTOLOGY = Path(__file__).resolve().parent.parent / "data" / "ontology.yaml"


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    return Ontology.load(ONTOLOGY)


def _doc(body: str, doc_id: str = "svc-fake", title: str = "Fake Service") -> Document:
    return Document(
        doc_id=doc_id,
        title=title,
        doc_type="service",
        date=None,
        body=body,
        path=Path(f"{doc_id}.txt"),
    )


def _edges(triples) -> set[tuple[str, str, str]]:
    return {(t.subject, t.predicate, t.object) for t in triples}


def test_a_forward_pattern_extracts_a_triple_carrying_its_source_sentence(ontology):
    sentence = "Atlas is owned by the Platform Team."
    triples = extract_triples([_doc(sentence)], ontology)

    assert len(triples) == 1
    triple = triples[0]
    assert (triple.subject, triple.predicate, triple.object) == (
        "Atlas",
        "owned_by",
        "Platform Team",
    )
    # Provenance is the whole point of the rule-based extractor: every edge has
    # to be auditable back to the sentence a human can read.
    assert triple.sentence == sentence
    assert triple.doc_id == "svc-fake"
    assert triple.pattern == "{s} is owned by {o}"


def test_an_inverse_pattern_stores_the_triple_the_other_way_round(ontology):
    triples = extract_triples([_doc("The Platform Team owns Atlas.")], ontology)
    assert _edges(triples) == {("Atlas", "owned_by", "Platform Team")}


def test_an_alias_is_resolved_to_its_canonical_entity(ontology):
    triples = extract_triples([_doc("Ember Auth is owned by the Security Team.")], ontology)
    assert _edges(triples) == {("Ember", "owned_by", "Security Team")}


def test_both_relations_in_a_multi_clause_sentence_are_found(ontology):
    # Straight from bug-903.txt. Two relations, one sentence, joined by "and".
    sentence = (
        "This was a cross-team fix, since Ember is owned by the Security Team "
        "and Marcus Chen is a member of the Platform Team; it proceeded under "
        "the cross-team exception with Tomas Oliveira as reviewer."
    )
    assert _edges(extract_triples([_doc(sentence)], ontology)) == {
        ("Ember", "owned_by", "Security Team"),
        ("Marcus Chen", "member_of", "Platform Team"),
    }


def test_a_sentence_with_no_relation_yields_nothing(ontology):
    assert extract_triples([_doc("Atlas keeps no durable state of its own.")], ontology) == []


# --- subject resolved from the document, not the sentence -------------------
#
# "The team owns Atlas" names no subject. The spec flags this resolution as the
# source of most mis-extractions, so it gets its own tests.

_PLATFORM = {"doc_id": "team-platform", "title": "Platform Team"}


def test_a_pattern_with_no_subject_slot_resolves_it_from_the_document(ontology):
    triples = extract_triples([_doc("The team owns Atlas.", **_PLATFORM)], ontology)
    assert _edges(triples) == {("Atlas", "owned_by", "Platform Team")}


def test_an_endpoint_resolved_from_the_document_is_flagged(ontology):
    # Flagged rather than silent: this resolution is the least trustworthy step
    # in the extractor, so an auditor needs to see which edges depended on it.
    triple = extract_triples([_doc("The team owns Atlas.", **_PLATFORM)], ontology)[0]
    assert triple.resolved_from_document is True


def test_an_edge_read_entirely_from_the_sentence_is_not_flagged(ontology):
    triple = extract_triples([_doc("Atlas is owned by the Platform Team.")], ontology)[0]
    assert triple.resolved_from_document is False


def test_a_document_with_no_recognisable_entity_skips_subjectless_patterns(ontology):
    # Better to drop the edge than to attach it to whatever entity happens to
    # be mentioned first in the body.
    triples = extract_triples(
        [_doc("The team owns Atlas.", doc_id="ref-glossary", title="Glossary")],
        ontology,
    )
    assert triples == []


def test_a_negated_clause_does_not_become_an_edge(ontology):
    # Verbatim from team-platform.txt, two sentences after "The team owns
    # Atlas." The corpus says the Platform Team does NOT own Ember, and an
    # extractor that misses the negation asserts the exact opposite.
    triples = extract_triples(
        [
            _doc(
                "It does not own Ember, despite Atlas depending on it, and this "
                "split has been a recurring source of friction.",
                **_PLATFORM,
            )
        ],
        ontology,
    )
    assert ("Ember", "owned_by", "Platform Team") not in _edges(triples)


# --- the extraction report --------------------------------------------------


def test_stats_count_triples_and_break_them_down_by_predicate(ontology):
    triples = extract_triples(
        [
            _doc(
                "Atlas is owned by the Platform Team. Atlas depends on Ember. "
                "Beacon depends on Cinder."
            )
        ],
        ontology,
    )
    stats = extraction_stats(triples, ontology)
    assert stats.triple_count == 3
    assert stats.per_predicate["owned_by"] == 1
    assert stats.per_predicate["depends_on"] == 2


def test_a_predicate_that_never_matched_is_reported_as_zero_not_omitted(ontology):
    # A relation whose patterns never fire is a defect in exactly the way an
    # orphan entity is. Omitting it from the census hides that; a zero shows it.
    stats = extraction_stats(
        extract_triples([_doc("Atlas is owned by the Platform Team.")], ontology),
        ontology,
    )
    assert stats.per_predicate["depends_on"] == 0
    assert set(stats.per_predicate) == set(ontology.relations)


def test_stats_compare_entities_linked_against_entities_declared(ontology):
    stats = extraction_stats(
        extract_triples([_doc("Atlas is owned by the Platform Team.")], ontology),
        ontology,
    )
    assert stats.entities_linked == {"Atlas", "Platform Team"}
    assert stats.entities_declared == len(ontology.entities)


def test_an_entity_with_no_edge_is_reported_as_an_orphan(ontology):
    # An orphan means a pattern failed to match real prose. The report has to
    # name them, because "34 of 39 entities linked" hides which five broke.
    stats = extraction_stats(
        extract_triples([_doc("Atlas is owned by the Platform Team.")], ontology),
        ontology,
    )
    assert "Ember" in stats.orphans
    assert "Atlas" not in stats.orphans


# --- prose details the ontology's templates do not spell out -----------------


def test_a_determiner_before_a_slot_does_not_block_the_match(ontology):
    # Verbatim from inc-2088.txt. The ontology declares the alias "Data
    # Retention Policy v3" but the prose writes "the Data Retention Policy v3",
    # and requiring the alias to sit flush against "triggered " loses the edge.
    triples = extract_triples(
        [
            _doc(
                "INC-2088 triggered the Data Retention Policy v3.",
                doc_id="inc-2088",
                title="INC-2088: Beacon Event Loss",
            )
        ],
        ontology,
    )
    assert _edges(triples) == {
        ("INC-2088", "triggered", "Data Retention Policy v3"),
    }


# --- sign-off authority ------------------------------------------------------
#
# Elena Vasquez is the orphan the spec warns about: her document phrases
# everything as "chairs", "has final sign-off on", "made the original decision
# to place", so none of the shipped relations reach her. These four sentences
# are verbatim from the corpus and span three documents, which is what makes
# signed_off_by worth a relation rather than a one-off pattern.

_SIGN_OFF = [
    (
        "pol-vendor-review",
        "Vendor Review Policy",
        "It was authored by Aisha Bello and requires sign-off from Elena Vasquez "
        "for any new vendor on a production path.",
        ("Vendor Review Policy", "signed_off_by", "Elena Vasquez"),
    ),
    (
        "vendor-kestrel",
        "Kestrel Analytics",
        "The contract runs three years and is currently in year two, with Elena "
        "Vasquez holding sign-off.",
        ("Kestrel Analytics", "signed_off_by", "Elena Vasquez"),
    ),
    (
        "vendor-northwind",
        "Northwind Cloud",
        "Elena Vasquez holds sign-off on the contract, which is renewed annually "
        "and which is currently in its third year.",
        ("Northwind Cloud", "signed_off_by", "Elena Vasquez"),
    ),
    (
        "person-elena-vasquez",
        "Elena Vasquez",
        "She has final sign-off on the Vendor Review Policy and on any contract "
        "with a vendor that sits on a production request path.",
        ("Vendor Review Policy", "signed_off_by", "Elena Vasquez"),
    ),
]


@pytest.mark.parametrize(
    "doc_id,title,sentence,edge", _SIGN_OFF, ids=[row[0] for row in _SIGN_OFF]
)
def test_sign_off_authority_is_extracted(ontology, doc_id, title, sentence, edge):
    triples = extract_triples([_doc(sentence, doc_id=doc_id, title=title)], ontology)
    assert edge in _edges(triples)


# --- domain/range type checking ---------------------------------------------


def test_an_edge_whose_endpoints_do_not_type_check_is_rejected(ontology):
    # Nonsense as prose, but exactly the shape a mis-signed inverse pattern
    # produces: led_by(Person, Team) where the ontology declares Team -> Person.
    assert extract_triples([_doc("The Platform Team leads Priya Raman.")], ontology) == []


def test_the_same_relation_in_the_declared_direction_survives(ontology):
    triples = extract_triples([_doc("Priya Raman leads the Platform Team.")], ontology)
    assert _edges(triples) == {("Platform Team", "led_by", "Priya Raman")}


# --- the step's acceptance criteria, against the real corpus ----------------

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def corpus_triples(ontology):
    return extract_triples(load_corpus(RAW), ontology)


_REQUIRED = [
    ("Atlas", "depends_on", "Ember"),
    ("Ember", "owned_by", "Security Team"),
    ("BUG-903", "affects", "Ember"),
    ("BUG-903", "fixed_by", "Marcus Chen"),
    ("INC-2088", "caused_by", "Cinder"),
    ("INC-2088", "triggered", "Data Retention Policy v3"),
]


@pytest.mark.parametrize("edge", _REQUIRED, ids=[" ".join(e) for e in _REQUIRED])
def test_the_named_edges_are_extracted_from_the_corpus(corpus_triples, edge):
    assert edge in _edges(corpus_triples)


def test_no_declared_entity_is_left_without_an_edge(corpus_triples, ontology):
    # Zero orphans is the bar. A name here means a pattern lost to real prose.
    assert extraction_stats(corpus_triples, ontology).orphans == []


def test_every_triple_quotes_a_sentence_from_its_own_document(corpus_triples):
    # Provenance you cannot check is not provenance. Bodies are whitespace-
    # collapsed because split_sentences undoes the corpus's ~95-column wrap.
    bodies = {d.doc_id: " ".join(d.body.split()) for d in load_corpus(RAW)}
    for triple in corpus_triples:
        assert triple.pattern.strip(), f"{triple} carries no pattern"
        assert triple.sentence in bodies[triple.doc_id], (
            f"{triple.subject} -{triple.predicate}-> {triple.object} cites a "
            f"sentence that is not in {triple.doc_id}"
        )


def test_no_predicate_has_the_same_edge_in_both_directions(corpus_triples):
    # Both directions of one functional predicate means one of them is a
    # mis-signed pattern -- the failure domain/range constraints exist to kill.
    edges = _edges(corpus_triples)
    assert sorted(e for e in edges if (e[2], e[1], e[0]) in edges) == []
