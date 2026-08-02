from __future__ import annotations

from dataclasses import dataclass

from core.corpus import Document, split_sentences
from kgraph.ontology import Entity, Gazetteer, Ontology

# Which documents talk about which entities. Both page builders need this, and
# it is the only place that decides what a page's "primary" source is.


@dataclass(frozen=True)
class SourceIndex:
    documents: tuple[Document, ...]
    mentions: dict[str, tuple[str, ...]]  # entity name -> doc ids, in corpus order
    primary: dict[str, str]  # entity name -> the document the page is built from

    def document(self, doc_id: str) -> Document:
        return next(d for d in self.documents if d.doc_id == doc_id)

    def of_type(self, *doc_types: str) -> tuple[Document, ...]:
        return tuple(d for d in self.documents if d.doc_type in doc_types)


def index_sources(documents: list[Document], ontology: Ontology, gazetteer: Gazetteer) -> SourceIndex:
    counts: dict[str, dict[str, int]] = {e.name: {} for e in ontology.entities}
    for document in documents:
        for mention in gazetteer.find(f"{document.title}\n{document.body}"):
            counts[mention.name][document.doc_id] = counts[mention.name].get(document.doc_id, 0) + 1

    return SourceIndex(
        documents=tuple(documents),
        mentions={name: tuple(docs) for name, docs in counts.items()},
        primary={
            name: _primary(name, docs, documents) for name, docs in counts.items() if docs
        },
    )


def _primary(name: str, docs: dict[str, int], documents: list[Document]) -> str:
    """The document a page is mostly written from.

    A document whose title names the entity wins outright -- `svc-atlas.txt` is
    the Atlas page's source even if some incident report mentions Atlas more
    often. Otherwise the most mentions win, and doc id breaks ties so that
    recompiling never reshuffles a page.
    """
    titles = {d.doc_id: d.title for d in documents}
    return min(
        docs,
        key=lambda doc_id: (
            0 if name.lower() in titles[doc_id].lower() else 1,
            -docs[doc_id],
            doc_id,
        ),
    )


def linkify(text: str, gazetteer: Gazetteer, skip: str | None = None) -> str:
    """Wrap every entity mention in `[[...]]`, using its canonical name.

    Right to left so earlier offsets stay valid. `skip` keeps a page from
    linking to itself, which lint would otherwise report as a self-reference
    and which reads badly anyway.
    """
    for mention in reversed(gazetteer.find(text)):
        if mention.name == skip:
            continue
        text = f"{text[: mention.start]}[[{mention.name}]]{text[mention.end :]}"
    return text


def first_sentence(document: Document) -> str:
    sentences = split_sentences(document.body)
    return sentences[0] if sentences else document.title


def mentioning_sentences(
    entity: Entity, sources: SourceIndex, gazetteer: Gazetteer, exclude: str, limit: int
) -> list[tuple[str, str]]:
    """`(doc_id, sentence)` for sentences about `entity` outside its own page source."""
    found: list[tuple[str, str]] = []
    for doc_id in sources.mentions[entity.name]:
        if doc_id == exclude:
            continue
        document = sources.document(doc_id)
        for sentence in split_sentences(document.body):
            if any(m.name == entity.name for m in gazetteer.find(sentence)):
                found.append((doc_id, sentence))
                break  # one sentence per document keeps the page readable
        if len(found) >= limit:
            break
    return found
