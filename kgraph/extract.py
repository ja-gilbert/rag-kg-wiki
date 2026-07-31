from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from core.corpus import Document, split_sentences
from kgraph.ontology import Entity, Gazetteer, Ontology, Pattern, Relation

_SLOT = re.compile(r"\{([so])\}")

# A slot must not begin mid-word: without this, "v3" would match inside "rev3"
# exactly as it would for the gazetteer. The trailing boundary comes from the
# alternation itself.
_NOT_MID_WORD = r"(?<![A-Za-z0-9])"

# Prose puts determiners in front of names -- "triggered the Data Retention
# Policy v3" -- while the ontology declares the bare alias. Absorbing an
# optional "the" in the slot itself is the general fix; the alternative is
# declaring a second "the X" alias for every entity and still missing the next
# case. The determiner is outside the capture group, so the name that reaches
# the gazetteer is still the bare alias.
_DETERMINER = r"(?:[Tt]he )?"


@dataclass(frozen=True)
class Triple:
    """One extracted edge, with everything needed to audit it.

    `sentence` and `pattern` are not debug decoration: being able to click an
    edge in the UI and see the prose it came from, and the rule that read it,
    is the property the whole knowledge-graph approach is selling.
    """

    subject: str
    predicate: str
    object: str
    doc_id: str
    sentence: str
    pattern: str
    # True when one endpoint came from the document's primary entity rather
    # than from the sentence -- see _document_entity.
    resolved_from_document: bool = False


def _compile(template: str, alternation: str) -> re.Pattern[str]:
    parts: list[str] = []
    last = 0
    for slot in _SLOT.finditer(template):
        parts.append(re.escape(template[last : slot.start()]))
        parts.append(
            f"{_NOT_MID_WORD}{_DETERMINER}(?P<{slot.group(1)}>{alternation})"
        )
        last = slot.end()
    parts.append(re.escape(template[last:]))
    return re.compile("".join(parts))


class _CompiledPattern:
    def __init__(self, relation: Relation, pattern: Pattern, alternation: str) -> None:
        self.relation = relation
        self.predicate = relation.predicate
        self.template = pattern.template
        self.inverse = pattern.inverse
        self.regex = _compile(pattern.template, alternation)
        # "The team owns {o}" names no subject; it has to come from the document.
        self.needs_document_subject = "{s}" not in pattern.template


def _compile_all(ontology: Ontology, gazetteer: Gazetteer) -> list[_CompiledPattern]:
    return [
        _CompiledPattern(relation, pattern, gazetteer.alternation)
        for relation in ontology.relations.values()
        for pattern in relation.patterns
    ]


def _document_entity(document: Document, gazetteer: Gazetteer) -> Entity | None:
    """The entity a document is *about*, taken from its title.

    The title and not the body: "BUG-903: Introspect endpoint accepts expired
    tokens" is about BUG-903, while its first body sentence mentions Ember just
    as prominently. Leftmost title mention wins, which also picks INC-2088 out
    of "INC-2088: Beacon Event Loss" rather than Beacon.
    """
    mentions = gazetteer.find(document.title)
    return gazetteer.canonical(mentions[0].surface) if mentions else None


def extract_triples(documents: list[Document], ontology: Ontology) -> list[Triple]:
    gazetteer = Gazetteer(ontology)
    compiled = _compile_all(ontology, gazetteer)

    # Keyed to collapse the same edge read twice out of one sentence by two
    # different patterns, while keeping the same edge asserted in two different
    # documents as two triples -- those are two independent pieces of evidence.
    seen: dict[tuple[str, str, str, str, str], Triple] = {}

    for document in documents:
        document_entity = _document_entity(document, gazetteer)
        for sentence in split_sentences(document.body):
            for rule in compiled:
                if rule.needs_document_subject and document_entity is None:
                    # Dropping the edge beats attaching it to whichever entity
                    # the body happens to mention first.
                    continue
                for match in rule.regex.finditer(sentence):
                    if rule.needs_document_subject:
                        subject_entity = document_entity
                    else:
                        subject_entity = gazetteer.canonical(match.group("s"))
                    object_entity = gazetteer.canonical(match.group("o"))
                    if rule.inverse:
                        subject_entity, object_entity = object_entity, subject_entity
                    if not rule.relation.type_checks(
                        subject_entity.type, object_entity.type
                    ):
                        continue
                    subject, object_ = subject_entity.name, object_entity.name
                    key = (
                        subject,
                        rule.predicate,
                        object_,
                        document.doc_id,
                        sentence,
                    )
                    seen.setdefault(
                        key,
                        Triple(
                            subject=subject,
                            predicate=rule.predicate,
                            object=object_,
                            doc_id=document.doc_id,
                            sentence=sentence,
                            pattern=rule.template,
                            resolved_from_document=rule.needs_document_subject,
                        ),
                    )

    return list(seen.values())


@dataclass(frozen=True)
class ExtractionStats:
    """What the extractor managed to read, and what it missed.

    `orphans` is the number that matters. An entity declared in the ontology
    with no edge means a pattern failed against real prose -- which is a bug in
    the ontology, never a licence to reword the corpus.
    """

    triple_count: int
    entities_declared: int
    entities_linked: set[str]
    per_predicate: dict[str, int]
    orphans: list[str]


def extraction_stats(triples: list[Triple], ontology: Ontology) -> ExtractionStats:
    linked = {t.subject for t in triples} | {t.object for t in triples}
    counts = Counter(t.predicate for t in triples)
    return ExtractionStats(
        triple_count=len(triples),
        entities_declared=len(ontology.entities),
        entities_linked=linked,
        # Every declared predicate, including the ones that matched nothing: a
        # relation whose patterns never fire is a defect in the same way an
        # orphan entity is, and omitting it from the census hides it. Descending,
        # so those zeros sort to the bottom where they are impossible to miss.
        per_predicate={
            predicate: counts.get(predicate, 0)
            for predicate, _ in sorted(
                ((p, counts.get(p, 0)) for p in ontology.relations),
                key=lambda row: -row[1],
            )
        },
        orphans=sorted(e.name for e in ontology.entities if e.name not in linked),
    )
