from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# A pattern template ending in this marker matches the surface form but emits
# the triple the other way round: "{s} owns {o}||inverse" reads "owns" off the
# page and stores owned_by(o, s).
_INVERSE_MARKER = "||inverse"


@dataclass(frozen=True)
class Entity:
    name: str
    type: str
    aliases: tuple[str, ...]

    @property
    def surface_forms(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass(frozen=True)
class Pattern:
    template: str
    inverse: bool


@dataclass(frozen=True)
class Relation:
    predicate: str
    label: str
    inverse_label: str
    # Entity types allowed at each end. A triple whose endpoints don't type-check
    # is rejected outright, which kills reversed edges from a mis-signed inverse
    # pattern -- "Platform Team owned_by Atlas" -- as a class rather than one at
    # a time. It cannot catch a reversal where domain and range overlap
    # (depends_on, supersedes); those need a different check.
    domain: tuple[str, ...]
    range: tuple[str, ...]
    patterns: tuple[Pattern, ...]

    def type_checks(self, subject_type: str, object_type: str) -> bool:
        return subject_type in self.domain and object_type in self.range


@dataclass(frozen=True)
class Mention:
    name: str
    type: str
    surface: str
    start: int
    end: int


def _parse_pattern(raw: str) -> Pattern:
    inverse = raw.endswith(_INVERSE_MARKER)
    template = raw[: -len(_INVERSE_MARKER)] if inverse else raw
    return Pattern(template=template.strip(), inverse=inverse)


class Ontology:
    """The one hand-authored input to graph construction.

    Declares what kinds of things exist, what they are called, and which surface
    forms in prose express which relation. Everything else is read off the raw
    text -- so this file is the only place a human gets to assert anything.
    """

    def __init__(
        self,
        entity_types: dict[str, Any],
        entities: list[Entity],
        relations: dict[str, Relation],
    ) -> None:
        self.entity_types = entity_types
        self.entities = entities
        self.relations = relations
        self._by_name = {e.name: e for e in entities}

    @classmethod
    def load(cls, path: Path | str) -> Ontology:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        entities = [
            Entity(
                name=row["name"],
                type=row["type"],
                aliases=tuple(row.get("aliases") or ()),
            )
            for row in raw["entities"]
        ]

        relations = {
            predicate: Relation(
                predicate=predicate,
                label=spec["label"],
                inverse_label=spec["inverse_label"],
                domain=tuple(spec["domain"]),
                range=tuple(spec["range"]),
                patterns=tuple(_parse_pattern(p) for p in spec["patterns"]),
            )
            for predicate, spec in raw["relations"].items()
        }

        return cls(raw["entity_types"], entities, relations)

    def entity(self, name: str) -> Entity:
        return self._by_name[name]


class Gazetteer:
    """Longest-alias-wins entity matching over raw sentences.

    One compiled alternation rather than a loop over aliases: the alternatives
    are ordered longest-first so that "Dr. Elena Vasquez" wins over the "Elena"
    sitting inside it, and a single left-to-right scan then yields
    non-overlapping mentions in document order for free.
    """

    def __init__(self, ontology: Ontology) -> None:
        self._canonical: dict[str, Entity] = {}
        for entity in ontology.entities:
            for surface in entity.surface_forms:
                self._canonical[surface] = entity

        alternatives = sorted(self._canonical, key=len, reverse=True)
        # \b on both ends keeps short aliases ("v2", "v3") from matching inside
        # a longer word. The trailing boundary is conditional because an alias
        # may end in a non-word character, where \b would never fire.
        # Exposed as source rather than a compiled object because the pattern
        # compiler in extract.py embeds it as the {s} and {o} slots.
        self.alternation = "|".join(
            re.escape(a) + (r"\b" if a[-1].isalnum() else "") for a in alternatives
        )
        self._pattern = re.compile(self.alternation)

    def canonical(self, surface: str) -> Entity:
        return self._canonical[surface]

    def find(self, text: str) -> list[Mention]:
        mentions = []
        for match in self._pattern.finditer(text):
            # A match starting mid-word is a coincidence, not a mention.
            if match.start() > 0 and text[match.start() - 1].isalnum():
                continue
            entity = self._canonical[match.group(0)]
            mentions.append(
                Mention(
                    name=entity.name,
                    type=entity.type,
                    surface=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
        return mentions
