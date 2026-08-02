from __future__ import annotations

import datetime as dt

from kgraph.graph import KnowledgeGraph
from kgraph.ontology import Entity, Gazetteer
from wikigen.page import WikiPage
from wikigen.sources import SourceIndex, first_sentence, linkify, mentioning_sentences

# One page per declared entity, compiled extractively: every sentence on the
# page came from a raw document, and every relationship bullet came from the
# knowledge graph. Nothing here paraphrases, which is what makes the output
# checkable against the corpus sentence by sentence.

_HISTORY_LIMIT = 4


def entity_page(
    entity: Entity,
    sources: SourceIndex,
    graph: KnowledgeGraph,
    gazetteer: Gazetteer,
    updated: dt.date,
) -> WikiPage:
    primary_id = sources.primary[entity.name]
    primary = sources.document(primary_id)

    # `cited` accumulates only what the page actually drew on. Citing every
    # document that merely mentions the entity would inflate provenance into
    # uselessness: a reader following one of those citations finds nothing the
    # page ever said.
    cited = [primary_id]

    sections = [("What it is", linkify(_opening(primary), gazetteer, skip=entity.name))]

    facts = sorted({(f.label, f.other, f.edge.doc_ids) for f in graph.facts_about(entity.name)})
    if facts:
        sections.append(
            ("Relationships", "\n".join(f"- {label} [[{other}]]" for label, other, _ in facts))
        )
        cited += [doc_id for *_, doc_ids in facts for doc_id in doc_ids]

    history = mentioning_sentences(
        entity, sources, gazetteer, exclude=primary_id, limit=_HISTORY_LIMIT
    )
    if history:
        sections.append(
            (
                "History",
                "\n".join(
                    f"- {linkify(sentence, gazetteer, skip=entity.name)} ({doc_id})"
                    for doc_id, sentence in history
                ),
            )
        )
        cited += [doc_id for doc_id, _ in history]

    related = sorted({other for _, other, _ in facts})
    if related:
        sections.append(("Related", " · ".join(f"[[{other}]]" for other in related)))

    return WikiPage(
        title=entity.name,
        page_type="entity",
        entity_type=entity.type,
        sources=tuple(dict.fromkeys(cited)),
        updated=updated,
        summary=first_sentence(primary),
        sections=tuple(sections),
    )


def _opening(document) -> str:
    """The first paragraph of the primary document, verbatim."""
    paragraph = document.body.split("\n\n", 1)[0]
    return " ".join(paragraph.split())
