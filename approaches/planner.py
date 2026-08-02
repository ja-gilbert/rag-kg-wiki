from __future__ import annotations

import re
from dataclasses import dataclass

from kgraph.ontology import Gazetteer, Ontology

# Turning a question into a graph query. Split from kg.py because this is the
# only part of the approach that knows any English -- everything downstream
# works in entities, predicates and hops.

# Irregular English only. Every other answer type falls out of the entity type
# names the ontology already declares ("which policy" -> Policy), so a new type
# is targetable the moment it is added, with no change here.
_INTERROGATIVE_TYPE = {
    "who": "Person",
    "whom": "Person",
    "whose": "Person",
}
_IRREGULAR_PLURAL = {"people": "Person"}

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

# Deliberately literal. A question either asks about graph distance in so many
# words or it does not, and anything that fails to match falls through to
# ordinary path search rather than to a guess.
_HOP_PATTERNS = (
    re.compile(r"\bwithin (\d+|one|two|three|four|five) hops?\b", re.IGNORECASE),
    re.compile(r"\b(\d+|one|two|three|four|five) (?:hops?|degrees) (?:from|of)\b", re.IGNORECASE),
)

_WORD = re.compile(r"[a-z0-9][a-z0-9-]*")


@dataclass(frozen=True)
class QuestionPlan:
    seeds: tuple[str, ...]
    answer_type: str | None
    named_types: tuple[str, ...]
    cues: tuple[str, ...]
    hops: int | None  # set only when the question asks a neighbourhood question


def plan_question(question: str, ontology: Ontology, gazetteer: Gazetteer) -> QuestionPlan:
    words = _WORD.findall(question.lower())
    named = _named_types(question, words, ontology)
    return QuestionPlan(
        seeds=tuple(dict.fromkeys(m.name for m in gazetteer.find(question))),
        answer_type=_answer_type(question, words, ontology, named),
        named_types=named,
        cues=_cues(words, ontology),
        hops=_hops(question),
    )


def _type_surfaces(type_name: str) -> set[str]:
    lowered = type_name.lower()
    return {lowered, f"{lowered}s"}


def _named_types(question: str, words: list[str], ontology: Ontology) -> tuple[str, ...]:
    present = set(words)
    named = [
        type_name
        for type_name in ontology.entity_types
        if _type_surfaces(type_name) & present
    ]
    named += [
        type_name
        for plural, type_name in _IRREGULAR_PLURAL.items()
        if plural in present and type_name not in named
    ]
    return tuple(named)


def _answer_type(
    question: str, words: list[str], ontology: Ontology, named: tuple[str, ...]
) -> str | None:
    # "which policy", "what service" -- the type attached to the interrogative
    # is the one being asked for, which matters in a question that names three.
    for n, word in enumerate(words[:-1]):
        if word in ("which", "what"):
            following = words[n + 1]
            for type_name in ontology.entity_types:
                if following in _type_surfaces(type_name):
                    return type_name
            if following in _IRREGULAR_PLURAL:
                return _IRREGULAR_PLURAL[following]

    for word in words:
        if word in _INTERROGATIVE_TYPE:
            return _INTERROGATIVE_TYPE[word]

    return named[0] if named else None


def _cues(words: list[str], ontology: Ontology) -> tuple[str, ...]:
    """Predicates the question's verbs point at.

    Derived from the relation names and labels the ontology already carries, so
    there is no second vocabulary to keep in step: `fixed_by` is cued by
    "fixed", `depends_on` by "depends". A relation becomes cueable the moment
    it is declared.
    """
    present = set(words)
    cues = []
    for predicate, relation in ontology.relations.items():
        surfaces = {predicate, predicate.replace("_", " "), relation.label, relation.inverse_label}
        # The leading word carries the verb: "fixed by" -> "fixed". Questions
        # inflect the rest away ("who fixed it", not "who fixed by it").
        leading = {s.split()[0].replace("_", " ").split()[0] for s in surfaces if s}
        if leading & present:
            cues.append(predicate)
    return tuple(cues)


def _hops(question: str) -> int | None:
    for pattern in _HOP_PATTERNS:
        match = pattern.search(question)
        if match:
            raw = match.group(1).lower()
            return _NUMBER_WORDS.get(raw) or int(raw)
    return None
