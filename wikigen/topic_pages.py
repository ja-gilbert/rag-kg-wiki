from __future__ import annotations

import datetime as dt

from kgraph.graph import KnowledgeGraph
from kgraph.ontology import Ontology
from wikigen.page import WikiPage
from wikigen.sources import SourceIndex, first_sentence

# The pages that beat RAG, because their content exists in no single source
# document. Each one is a handful of graph queries laid out as prose -- the
# synthesis happens once, here, instead of on every query the way RAG does it.

# Which document types a topic answers for. Used to place any document that no
# entity page happened to cite, so the corpus stays fully covered.
TOPIC_DOC_TYPES = {
    "Architecture Overview": ("service", "reference"),
    "Incident History": ("incident", "bug"),
    "Vendor Risk": ("vendor", "policy"),
    "Who Owns What": ("team", "person", "runbook"),
}


def topic_pages(
    graph: KnowledgeGraph, sources: SourceIndex, ontology: Ontology, updated: dt.date
) -> list[WikiPage]:
    return [
        _architecture(graph, sources, ontology, updated),
        _incidents(graph, sources, ontology, updated),
        _vendors(graph, sources, ontology, updated),
        _ownership(graph, sources, ontology, updated),
    ]


# --- builders ----------------------------------------------------------------


def _architecture(graph, sources, ontology, updated) -> WikiPage:
    services = _named(ontology, "Service")
    return _page(
        title="Architecture Overview",
        summary="How the five services fit together, what they run on, and who owns each one.",
        sections=(
            ("The services", _catalogue(services, sources)),
            ("What depends on what", _edges(graph, "depends_on")),
            ("Where it runs", _edges(graph, "runs_on")),
            ("Who owns it", _edges(graph, "owned_by")),
        ),
        entities=services,
        sources_index=sources,
        updated=updated,
    )


def _incidents(graph, sources, ontology, updated) -> WikiPage:
    incidents = _named(ontology, "Incident")
    bugs = _named(ontology, "Bug")
    return _page(
        title="Incident History",
        summary="Every incident and bug in the corpus, what caused each one, and what it changed.",
        sections=(
            ("Incidents", _catalogue(incidents, sources)),
            ("What caused them", _edges(graph, "caused_by")),
            ("What they changed", _edges(graph, "triggered")),
            ("Bugs", _catalogue(bugs, sources)),
            ("What the bugs affected", _edges(graph, "affects")),
        ),
        entities=incidents + bugs,
        sources_index=sources,
        updated=updated,
    )


def _vendors(graph, sources, ontology, updated) -> WikiPage:
    vendors = _named(ontology, "Vendor")
    policies = _named(ontology, "Policy")
    return _page(
        title="Vendor Risk",
        summary="What has repeatedly gone wrong with our vendors, and what we changed because of it.",
        sections=(
            ("The vendors", _catalogue(vendors, sources)),
            ("What went wrong", _edges(graph, "caused_by")),
            ("The policies that followed", _catalogue(policies, sources)),
            ("What supersedes what", _edges(graph, "supersedes")),
        ),
        entities=vendors + policies,
        sources_index=sources,
        updated=updated,
    )


def _ownership(graph, sources, ontology, updated) -> WikiPage:
    teams = _named(ontology, "Team")
    people = _named(ontology, "Person")
    return _page(
        title="Who Owns What",
        summary="Which team owns which service, who leads each team, and who belongs to it.",
        sections=(
            ("The teams", _catalogue(teams, sources)),
            ("Service ownership", _edges(graph, "owned_by")),
            ("Who leads", _edges(graph, "led_by")),
            ("Who belongs where", _edges(graph, "member_of")),
        ),
        entities=teams + people,
        sources_index=sources,
        updated=updated,
    )


# --- shared ------------------------------------------------------------------


def _named(ontology: Ontology, entity_type: str) -> list[str]:
    return sorted(e.name for e in ontology.entities if e.type == entity_type)


def _catalogue(names: list[str], sources: SourceIndex) -> str:
    lines = []
    for name in names:
        primary = sources.primary.get(name)
        summary = first_sentence(sources.document(primary)) if primary else ""
        lines.append(f"- [[{name}]] — {summary}" if summary else f"- [[{name}]]")
    return "\n".join(lines)


def _edges(graph: KnowledgeGraph, predicate: str) -> tuple[str, list[str]]:
    """The rendered bullets, and the documents those edges were extracted from."""
    if predicate not in graph.ontology.relations:
        return "", []
    label = graph.ontology.relations[predicate].label
    edges = sorted(graph.match(predicate=predicate), key=lambda e: (e.subject, e.object))
    rendered = "\n".join(f"- [[{e.subject}]] {label} [[{e.object}]]" for e in edges)
    return rendered, [doc_id for e in edges for doc_id in e.doc_ids]


def _page(
    title: str,
    summary: str,
    sections: tuple[tuple[str, str | tuple[str, list[str]]], ...],
    entities: list[str],
    sources_index: SourceIndex,
    updated: dt.date,
) -> WikiPage:
    # A topic page's claim to being synthesis is only as good as its
    # provenance, so it cites what it was built from: the primary document
    # behind each entity it catalogues, and the documents behind each edge it
    # lists. Not every document that happens to mention them.
    cited = [
        sources_index.primary[name] for name in entities if name in sources_index.primary
    ]
    resolved = []
    for heading, body in sections:
        if isinstance(body, tuple):
            body, doc_ids = body
            cited += doc_ids
        resolved.append((heading, body))
    sections = tuple(resolved)

    return WikiPage(
        title=title,
        page_type="topic",
        entity_type=None,
        sources=tuple(sorted(set(cited))),
        updated=updated,
        summary=summary,
        sections=tuple((heading, body) for heading, body in sections if body),
    )
