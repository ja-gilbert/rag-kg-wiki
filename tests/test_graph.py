from __future__ import annotations

from pathlib import Path as FilePath

import pytest

from core.corpus import load_corpus
from kgraph.extract import Triple, extract_triples
from kgraph.graph import KnowledgeGraph
from kgraph.ontology import Ontology

ONTOLOGY = FilePath(__file__).resolve().parent.parent / "data" / "ontology.yaml"
RAW = FilePath(__file__).resolve().parent.parent / "data" / "raw"


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


def _graph(triples: list[Triple], ontology: Ontology) -> KnowledgeGraph:
    return KnowledgeGraph(triples, ontology)


# --- construction ------------------------------------------------------------


def test_a_triple_becomes_an_edge_between_its_endpoints(ontology):
    graph = _graph([_triple("Atlas", "depends_on", "Ember")], ontology)

    assert set(graph.nodes) == {"Atlas", "Ember"}
    edge = graph.edge("Atlas", "depends_on", "Ember")
    assert (edge.subject, edge.predicate, edge.object) == ("Atlas", "depends_on", "Ember")


def test_two_different_relations_between_one_pair_both_survive(ontology):
    # The reason the store is a MultiDiGraph. Collapsing to a DiGraph would keep
    # whichever of these was inserted last and silently lose the other.
    graph = _graph(
        [
            _triple("BUG-903", "affects", "Ember"),
            _triple("BUG-903", "reported_by", "Marcus Chen"),
            _triple("BUG-903", "fixed_by", "Marcus Chen"),
        ],
        ontology,
    )

    predicates = {e.predicate for e in graph.match(subject="BUG-903", object="Marcus Chen")}
    assert predicates == {"reported_by", "fixed_by"}


def test_one_edge_asserted_in_two_documents_keeps_both_sources(ontology):
    # Two documents independently asserting the same fact is two pieces of
    # evidence for one edge, not two edges -- otherwise every path through it
    # would be enumerated twice with identical nodes.
    graph = _graph(
        [
            _triple("Atlas", "depends_on", "Ember", doc_id="svc-atlas"),
            _triple("Atlas", "depends_on", "Ember", doc_id="arch-overview"),
        ],
        ontology,
    )

    edge = graph.edge("Atlas", "depends_on", "Ember")
    assert {s.doc_id for s in edge.sources} == {"svc-atlas", "arch-overview"}
    assert len(graph.edges) == 1


# --- triple-pattern matching with wildcards ----------------------------------

_MATCHABLE = [
    _triple("Atlas", "owned_by", "Platform Team"),
    _triple("Beacon", "owned_by", "Platform Team"),
    _triple("Ember", "owned_by", "Security Team"),
    _triple("Atlas", "depends_on", "Ember"),
]


def test_a_fully_bound_pattern_matches_just_that_edge(ontology):
    matched = _graph(_MATCHABLE, ontology).match(
        subject="Atlas", predicate="owned_by", object="Platform Team"
    )
    assert [(e.subject, e.predicate, e.object) for e in matched] == [
        ("Atlas", "owned_by", "Platform Team")
    ]


def test_an_unbound_subject_matches_every_edge_with_that_predicate_and_object(ontology):
    matched = _graph(_MATCHABLE, ontology).match(
        predicate="owned_by", object="Platform Team"
    )
    assert {e.subject for e in matched} == {"Atlas", "Beacon"}


def test_an_unbound_pattern_matches_the_whole_graph(ontology):
    assert len(_graph(_MATCHABLE, ontology).match()) == len(_MATCHABLE)


def test_a_pattern_that_matches_nothing_returns_empty(ontology):
    assert _graph(_MATCHABLE, ontology).match(subject="Cinder") == []


# --- facts_about -------------------------------------------------------------


def test_facts_about_an_entity_include_edges_where_it_is_the_object(ontology):
    # Ember does not appear as the subject of either of these, but "what is
    # affected by BUG-903" is a fact about Ember by any useful definition.
    graph = _graph(
        [
            _triple("Ember", "owned_by", "Security Team"),
            _triple("BUG-903", "affects", "Ember"),
        ],
        ontology,
    )

    facts = graph.facts_about("Ember")
    assert {(f.subject, f.predicate, f.object) for f in facts} == {
        ("Ember", "owned_by", "Security Team"),
        ("BUG-903", "affects", "Ember"),
    }


def test_a_fact_reads_from_the_perspective_of_the_entity_asked_about(ontology):
    # Rendered for a UI panel headed "Ember", so the inbound edge has to phrase
    # itself with the relation's inverse label rather than read backwards.
    graph = _graph([_triple("BUG-903", "affects", "Ember")], ontology)

    (fact,) = graph.facts_about("Ember")
    assert fact.render() == "Ember affected by BUG-903"


def test_an_outbound_fact_reads_in_the_declared_direction(ontology):
    graph = _graph([_triple("Ember", "owned_by", "Security Team")], ontology)

    (fact,) = graph.facts_about("Ember")
    assert fact.render() == "Ember owned by Security Team"


def test_every_fact_carries_the_sentence_it_came_from(ontology):
    graph = _graph([_triple("Ember", "owned_by", "Security Team")], ontology)

    (fact,) = graph.facts_about("Ember")
    assert fact.sentence == "Ember owned by Security Team."
    assert fact.doc_id == "svc-fake"


# --- neighbourhood queries ---------------------------------------------------

_NEIGHBOURHOOD = [
    _triple("INC-2041", "affects", "Atlas"),
    _triple("INC-2041", "resolved_by", "Lin Zhao"),
    _triple("Atlas", "owned_by", "Platform Team"),
    _triple("Platform Team", "led_by", "Priya Raman"),
    _triple("Marcus Chen", "member_of", "Platform Team"),
]


def test_neighbours_within_one_hop_are_the_direct_endpoints(ontology):
    found = _graph(_NEIGHBOURHOOD, ontology).neighbors_within("INC-2041", hops=1)
    assert {n.name for n in found} == {"Atlas", "Lin Zhao"}


def test_the_starting_entity_is_not_its_own_neighbour(ontology):
    found = _graph(_NEIGHBOURHOOD, ontology).neighbors_within("INC-2041", hops=2)
    assert "INC-2041" not in {n.name for n in found}


def test_traversal_walks_edges_against_their_direction(ontology):
    # Nothing points out of Atlas towards INC-2041 -- the edge runs the other
    # way. A directed-only walk would report Atlas as having no incident.
    found = _graph(_NEIGHBOURHOOD, ontology).neighbors_within("Atlas", hops=1)
    assert "INC-2041" in {n.name for n in found}


def test_neighbours_can_be_filtered_to_one_entity_type(ontology):
    # "Which people are within two hops of INC-2041?" -- the demo question that
    # has no textual answer anywhere in the corpus. Atlas and the Platform Team
    # are both inside the radius and must still be filtered out of the answer.
    graph = _graph(_NEIGHBOURHOOD, ontology)
    assert {n.name for n in graph.neighbors_within("INC-2041", hops=2)} == {
        "Atlas",
        "Lin Zhao",
        "Platform Team",
    }
    found = graph.neighbors_within("INC-2041", hops=2, types=("Person",))
    assert {n.name for n in found} == {"Lin Zhao"}


def test_the_type_filter_does_not_block_traversal_through_other_types(ontology):
    # Priya Raman is only reachable via Atlas and the Platform Team. Filtering
    # the *results* to Person must not stop the walk from crossing non-Person
    # nodes on the way, which is the obvious way to get this wrong.
    found = _graph(_NEIGHBOURHOOD, ontology).neighbors_within(
        "INC-2041", hops=3, types=("Person",)
    )
    assert "Priya Raman" in {n.name for n in found}


def test_a_neighbour_reports_how_far_away_it_is(ontology):
    found = {
        n.name: n.distance
        for n in _graph(_NEIGHBOURHOOD, ontology).neighbors_within("INC-2041", hops=3)
    }
    assert found["Atlas"] == 1
    assert found["Platform Team"] == 2
    assert found["Priya Raman"] == 3


def test_a_neighbour_carries_the_path_that_reached_it(ontology):
    # Proximity in a graph is a fact rather than a vibe, but only if you can see
    # the route -- otherwise "within two hops" is as unfalsifiable as a score.
    (neighbour,) = [
        n
        for n in _graph(_NEIGHBOURHOOD, ontology).neighbors_within("INC-2041", hops=2)
        if n.name == "Platform Team"
    ]
    assert neighbour.via.nodes == ("INC-2041", "Atlas", "Platform Team")


def test_an_unknown_entity_has_no_neighbourhood(ontology):
    assert _graph(_NEIGHBOURHOOD, ontology).neighbors_within("Nobody", hops=2) == []


# --- paths_between -----------------------------------------------------------

_ROUTE = [
    _triple("Atlas", "depends_on", "Ember", doc_id="svc-atlas"),
    _triple("BUG-903", "affects", "Ember", doc_id="bug-903"),
    _triple("BUG-903", "fixed_by", "Marcus Chen", doc_id="bug-903"),
]


def test_paths_between_finds_the_route_through_a_reversed_edge(ontology):
    (path,) = _graph(_ROUTE, ontology).paths_between("Atlas", "Marcus Chen", max_hops=4)
    assert path.nodes == ("Atlas", "Ember", "BUG-903", "Marcus Chen")


def test_max_hops_bounds_the_search(ontology):
    assert _graph(_ROUTE, ontology).paths_between("Atlas", "Marcus Chen", max_hops=2) == []


def test_two_unconnected_entities_have_no_path(ontology):
    graph = _graph([_triple("Atlas", "depends_on", "Ember")], ontology)
    assert graph.paths_between("Atlas", "Marcus Chen", max_hops=4) == []


def test_an_unknown_entity_has_no_path_rather_than_raising(ontology):
    # The graph's honest answer to a question about something it has never heard
    # of. Step 6 turns this into "no path", so it must not be an exception.
    graph = _graph(_ROUTE, ontology)
    assert graph.paths_between("Atlas", "Meridian Systems' revenue", max_hops=4) == []


def test_paths_are_returned_shortest_first(ontology):
    graph = _graph(
        _ROUTE + [_triple("Atlas", "reported_by", "Marcus Chen", doc_id="svc-atlas")],
        ontology,
    )
    lengths = [len(p.hops) for p in graph.paths_between("Atlas", "Marcus Chen", max_hops=4)]
    assert lengths == sorted(lengths)
    assert lengths[0] == 1


def test_a_path_never_revisits_a_node(ontology):
    for path in _graph(_ROUTE, ontology).paths_between("Atlas", "Marcus Chen", max_hops=4):
        assert len(set(path.nodes)) == len(path.nodes)


# --- path rendering ----------------------------------------------------------


def test_a_path_renders_each_edge_in_its_real_direction(ontology):
    # The arrow flips rather than the label: BUG-903 affects Ember, and a path
    # that walked it backwards has to say so instead of inventing "Ember affects
    # BUG-903". This string is the demo's single most explanatory artefact.
    (path,) = _graph(_ROUTE, ontology).paths_between("Atlas", "Marcus Chen", max_hops=4)
    assert path.render() == (
        "Atlas -[depends on]-> Ember <-[affects]- BUG-903 -[fixed by]-> Marcus Chen"
    )


def test_every_hop_carries_the_sentence_that_justifies_it(ontology):
    (path,) = _graph(_ROUTE, ontology).paths_between("Atlas", "Marcus Chen", max_hops=4)
    assert [h.sentence for h in path.hops] == [
        "Atlas depends on Ember.",
        "BUG-903 affects Ember.",
        "BUG-903 fixed by Marcus Chen.",
    ]
    assert [h.doc_id for h in path.hops] == ["svc-atlas", "bug-903", "bug-903"]


# --- serialisation for the graph visualisation -------------------------------


def test_every_serialised_node_carries_its_ontology_type(ontology):
    # The frontend colours nodes by entity type, so the type has to survive the
    # trip rather than be re-derived from the name in JavaScript.
    payload = _graph(_ROUTE, ontology).to_dict()
    types = {n["id"]: n["type"] for n in payload["nodes"]}
    assert types == {
        "Atlas": "Service",
        "Ember": "Service",
        "BUG-903": "Bug",
        "Marcus Chen": "Person",
    }


def test_every_serialised_edge_carries_its_label_and_provenance(ontology):
    payload = _graph(_ROUTE, ontology).to_dict()
    edge = next(e for e in payload["edges"] if e["predicate"] == "affects")
    assert (edge["source"], edge["target"]) == ("BUG-903", "Ember")
    assert edge["label"] == "affects"
    assert edge["sentence"] == "BUG-903 affects Ember."
    assert edge["doc_id"] == "bug-903"


def test_the_serialisation_is_json_round_trippable(ontology):
    import json

    payload = _graph(_ROUTE, ontology).to_dict()
    assert json.loads(json.dumps(payload)) == payload


# --- the step's acceptance criteria, against the real corpus -----------------


@pytest.fixture(scope="module")
def corpus_graph(ontology) -> KnowledgeGraph:
    return KnowledgeGraph(extract_triples(load_corpus(RAW), ontology), ontology)


def test_the_bug_903_route_is_found_in_the_real_corpus(corpus_graph):
    paths = corpus_graph.paths_between("Atlas", "Marcus Chen", max_hops=4)
    assert any(
        p.nodes == ("Atlas", "Ember", "BUG-903", "Marcus Chen") for p in paths
    ), [p.render() for p in paths]


def test_the_retention_policy_route_is_found_in_the_real_corpus(corpus_graph):
    paths = corpus_graph.paths_between("Beacon", "Data Retention Policy v3", max_hops=4)
    assert any(
        p.nodes == ("Beacon", "Cinder", "INC-2088", "Data Retention Policy v3")
        for p in paths
    ), [p.render() for p in paths]


def test_the_corpus_graph_is_one_connected_component(corpus_graph):
    # A second component means a group of entities nothing in the corpus relates
    # to the rest -- which is an extraction gap, not a fact about the company.
    assert corpus_graph.component_count() == 1, corpus_graph.components()[1:]


def test_every_hop_of_every_corpus_path_quotes_a_sentence(corpus_graph):
    paths = corpus_graph.paths_between("Atlas", "Marcus Chen", max_hops=4)
    assert paths
    for path in paths:
        for hop in path.hops:
            assert hop.sentence.strip(), f"{path.render()} has an unjustified hop"
            assert hop.doc_id.strip()


def test_people_within_two_hops_of_inc_2041_is_answerable(corpus_graph):
    # The demo question with no textual answer anywhere in the corpus.
    found = corpus_graph.neighbors_within("INC-2041", hops=2, types=("Person",))
    assert found
    assert all(n.distance <= 2 for n in found)
