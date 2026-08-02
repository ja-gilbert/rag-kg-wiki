from __future__ import annotations

from pathlib import Path

import pytest

from approaches.kg import KgApproach
from core.corpus import load_corpus
from core.llm import ExtractiveBackend
from kgraph.extract import Triple, extract_triples
from kgraph.graph import KnowledgeGraph
from kgraph.ontology import Ontology

ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY = ROOT / "data" / "ontology.yaml"
RAW = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    return Ontology.load(ONTOLOGY)


def _triple(subject: str, predicate: str, object_: str, doc_id: str = "svc-fake") -> Triple:
    sentence = f"{subject} {predicate.replace('_', ' ')} {object_}."
    return Triple(
        subject=subject,
        predicate=predicate,
        object=object_,
        doc_id=doc_id,
        sentence=sentence,
        pattern="{s} " + predicate.replace("_", " ") + " {o}",
    )


# Atlas reaches Marcus Chen two ways: the three-hop route the question is
# actually about, and a two-hop shortcut through the team. Scoring has to
# prefer the longer one, or the planner is just breadth-first search.
_ROUTES = [
    _triple("Atlas", "depends_on", "Ember", doc_id="svc-atlas"),
    _triple("BUG-903", "affects", "Ember", doc_id="bug-903"),
    _triple("BUG-903", "fixed_by", "Marcus Chen", doc_id="bug-903"),
    _triple("Atlas", "owned_by", "Platform Team", doc_id="svc-atlas"),
    _triple("Marcus Chen", "member_of", "Platform Team", doc_id="team-platform"),
]

_THE_QUESTION = "Who fixed the bug in the service that Atlas depends on?"


def _kg(ontology, triples=None, llm=None, **overrides) -> KgApproach:
    cfg = {"max_hops": 4, "max_answers": 6}
    return KgApproach(
        graph=KnowledgeGraph(_ROUTES if triples is None else triples, ontology),
        ontology=ontology,
        llm=llm or ExtractiveBackend(),
        cfg={**cfg, **overrides},
    )


class _RefusingBackend:
    """Fails loudly if asked to generate. Proves the no-path path stays offline."""

    name = "extractive"

    def generate(self, question, evidence):
        raise AssertionError("the generator must not be called when there is no evidence")


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def test_seeds_come_from_the_gazetteer(ontology):
    assert _kg(ontology).answer(_THE_QUESTION).detail["seeds"] == ["Atlas"]


def test_who_targets_a_person(ontology):
    assert _kg(ontology).answer(_THE_QUESTION).detail["answer_type"] == "Person"


def test_an_entity_type_named_in_the_question_becomes_the_target(ontology):
    plan = _kg(ontology).answer("Which policy did Atlas trigger?").detail
    assert plan["answer_type"] == "Policy"


def test_predicate_cues_are_derived_from_the_ontology(ontology):
    cues = _kg(ontology).answer(_THE_QUESTION).detail["cues"]
    assert "fixed_by" in cues
    assert "depends_on" in cues


def test_a_question_naming_no_known_entity_says_so(ontology):
    answer = _kg(ontology, llm=_RefusingBackend()).answer("What is our revenue?")
    assert answer.confident is False
    assert answer.note
    assert answer.detail["seeds"] == []


# --------------------------------------------------------------------------
# search and scoring
# --------------------------------------------------------------------------


def test_the_cue_matching_route_beats_the_shorter_one(ontology):
    answer = _kg(ontology).answer(_THE_QUESTION)
    assert answer.detail["answer_entity"] == "Marcus Chen"
    assert answer.detail["path"]["nodes"] == ["Atlas", "Ember", "BUG-903", "Marcus Chen"]


def test_the_winning_path_is_rendered_with_real_directions(ontology):
    rendered = _kg(ontology).answer(_THE_QUESTION).detail["path"]["render"]
    assert rendered == (
        "Atlas -[depends on]-> Ember <-[affects]- BUG-903 -[fixed by]-> Marcus Chen"
    )


def test_the_score_follows_the_documented_weights(ontology):
    # two distinct cues matched (+5.0), two intermediate types the question
    # named (+3.0), three hops (-1.8).
    assert _kg(ontology).answer(_THE_QUESTION).detail["score"] == pytest.approx(6.2)


def test_repeating_one_cue_does_not_outscore_matching_two(ontology):
    # A chain of three `depends on` edges explains one word of the question
    # three times over; the route the question describes explains two. Per-edge
    # scoring gets this backwards -- see KgApproach._score.
    triples = _ROUTES + [
        _triple("Ember", "depends_on", "Delta Store", doc_id="svc-ember"),
        _triple("Delta Store", "owned_by", "Data Team", doc_id="svc-delta-store"),
        _triple("Priya Raman", "member_of", "Data Team", doc_id="team-data"),
    ]
    answer = _kg(ontology, triples=triples).answer(_THE_QUESTION)
    assert answer.detail["answer_entity"] == "Marcus Chen"


def test_max_hops_bounds_the_search(ontology):
    answer = _kg(ontology, llm=_RefusingBackend(), max_hops=1).answer(_THE_QUESTION)
    assert answer.confident is False


def test_max_answers_bounds_the_candidate_list(ontology):
    answer = _kg(ontology, max_answers=1).answer(_THE_QUESTION)
    assert len(answer.detail["candidates"]) == 1


def test_runners_up_stay_visible_with_their_scores(ontology):
    candidates = _kg(ontology).answer("Who is connected to Atlas?").detail["candidates"]
    assert all("score" in c and "entity" in c for c in candidates)
    assert [c["score"] for c in candidates] == sorted((c["score"] for c in candidates), reverse=True)


# --------------------------------------------------------------------------
# no path
# --------------------------------------------------------------------------


def test_no_path_means_no_generation(ontology):
    # Decision: with no evidence there is nothing to generate from, and handing
    # an empty prompt to a live model is exactly how a graph demo starts
    # hallucinating.
    answer = _kg(ontology, llm=_RefusingBackend()).answer("Which policy covers Atlas?")
    assert answer.confident is False
    assert answer.evidence == []


def test_the_no_path_note_names_what_it_did_find(ontology):
    note = _kg(ontology, llm=_RefusingBackend()).answer("Which policy covers Atlas?").note
    assert "Atlas" in note


# --------------------------------------------------------------------------
# evidence and provenance
# --------------------------------------------------------------------------


def test_evidence_is_one_block_per_path(ontology):
    answer = _kg(ontology).answer(_THE_QUESTION)
    assert len(answer.evidence) == len(answer.detail["candidates"])


def test_every_hop_of_the_evidence_quotes_its_sentence(ontology):
    hops = _kg(ontology).answer(_THE_QUESTION).evidence[0]["hops"]
    assert [h["sentence"] for h in hops] == [
        "Atlas depends on Ember.",
        "BUG-903 affects Ember.",
        "BUG-903 fixed by Marcus Chen.",
    ]


def test_citations_are_every_document_the_path_crossed(ontology):
    assert _kg(ontology).answer(_THE_QUESTION).citations == ["svc-atlas", "bug-903"]


def test_the_answer_names_the_entity_the_graph_arrived_at(ontology):
    assert "Marcus Chen" in _kg(ontology).answer(_THE_QUESTION).answer


# --------------------------------------------------------------------------
# neighbourhood queries
# --------------------------------------------------------------------------


def test_a_hop_question_routes_to_the_neighbourhood_query(ontology):
    answer = _kg(ontology).answer("Which people are within two hops of BUG-903?")
    assert answer.detail["query"] == "neighbourhood"
    assert answer.detail["hops"] == 2


def test_the_neighbourhood_is_filtered_to_the_answer_type(ontology):
    answer = _kg(ontology).answer("Which people are within two hops of BUG-903?")
    assert [n["entity"] for n in answer.detail["neighbours"]] == ["Marcus Chen"]


def test_a_neighbour_carries_the_route_that_reached_it(ontology):
    neighbour = _kg(ontology).answer(
        "Which people are within two hops of BUG-903?"
    ).detail["neighbours"][0]
    assert neighbour["distance"] == 1
    assert "Marcus Chen" in neighbour["via"]


def test_an_ordinary_question_is_not_a_neighbourhood_query(ontology):
    assert _kg(ontology).answer(_THE_QUESTION).detail["query"] == "path"


# --------------------------------------------------------------------------
# the envelope
# --------------------------------------------------------------------------


def test_the_answer_identifies_itself_as_kg(ontology):
    answer = _kg(ontology).answer(_THE_QUESTION)
    assert answer.approach == "kg"
    assert answer.label


def test_the_trace_says_what_it_did_in_plain_english(ontology):
    trace = _kg(ontology).answer(_THE_QUESTION).trace
    assert trace and all(isinstance(step, str) and step.strip() for step in trace)


def test_the_timing_covers_planning_and_search(ontology):
    ms = _kg(ontology).answer(_THE_QUESTION).ms
    assert {"plan", "search", "generate", "total"} <= set(ms)


# --------------------------------------------------------------------------
# against the real corpus
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_kg(ontology) -> KgApproach:
    graph = KnowledgeGraph(extract_triples(load_corpus(RAW), ontology), ontology)
    return KgApproach(
        graph=graph, ontology=ontology, llm=ExtractiveBackend(), cfg={"max_hops": 4, "max_answers": 6}
    )


def test_the_atlas_question_arrives_at_marcus_chen(corpus_kg):
    answer = corpus_kg.answer(_THE_QUESTION)
    assert answer.detail["answer_entity"] == "Marcus Chen"
    assert answer.detail["path"]["nodes"] == ["Atlas", "Ember", "BUG-903", "Marcus Chen"]
    assert answer.confident is True


def test_the_beacon_question_arrives_at_the_retention_policy(corpus_kg):
    answer = corpus_kg.answer(
        "Which policy came out of the incident caused by the service that Beacon depends on?"
    )
    assert answer.detail["answer_entity"] == "Data Retention Policy v3"
    assert answer.detail["path"]["nodes"] == [
        "Beacon",
        "Cinder",
        "INC-2088",
        "Data Retention Policy v3",
    ]


def test_the_beacon_question_does_not_route_through_to_the_superseded_policy(corpus_kg):
    # v3 supersedes v2, and a route to v3 extends one hop further to v2. Paying
    # the named-type bonus for a Policy waypoint made that longer route win,
    # which is the corpus's planted superseded-reference trap doing its job.
    answer = corpus_kg.answer(
        "Which policy came out of the incident caused by the service that Beacon depends on?"
    )
    assert "v2" not in answer.detail["answer_entity"]


def test_the_revenue_question_gets_an_explicit_no_path(corpus_kg):
    # The other half of non-negotiable #7: RAG answers this confidently at
    # cosine 0.47 (see tests/test_rag.py); the graph declines.
    answer = corpus_kg.answer("What was Meridian Systems' revenue last quarter?")
    assert answer.confident is False
    assert answer.evidence == []
    assert answer.note


def test_the_two_hop_question_is_answerable(corpus_kg):
    answer = corpus_kg.answer("Which people are within two hops of INC-2041?")
    assert answer.detail["query"] == "neighbourhood"
    assert answer.detail["neighbours"]
    assert all(n["distance"] <= 2 for n in answer.detail["neighbours"])
