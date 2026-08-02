from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

import yaml

# The page format is fixed because the query layer parses it back. Sections are
# held as raw (heading, body) pairs rather than as typed fields: that makes the
# round-trip trivially lossless, and it means a page whose Relationships block
# is malformed still parses -- reporting that is lint's job, not the parser's.

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_LINK = re.compile(r"\[\[([^\[\]]+)\]\]")
_RELATIONSHIP = re.compile(r"^- (.+?) \[\[([^\[\]]+)\]\]\s*$")
_HEADING = re.compile(r"^## (.+)$")

# Rendered from the `sources` field, so it is dropped on the way back in rather
# than being kept twice and allowed to disagree with the frontmatter.
_DERIVED_SECTION = "Sources"


@dataclass(frozen=True)
class WikiPage:
    title: str
    page_type: str  # "entity" | "topic"
    sources: tuple[str, ...]
    updated: dt.date
    summary: str
    sections: tuple[tuple[str, str], ...] = field(default=())
    entity_type: str | None = None

    # --- what the query layer and lint read ---------------------------------

    @property
    def links(self) -> tuple[str, ...]:
        """Every `[[target]]` on the page, in order, without repeats."""
        found = _LINK.findall(self.summary)
        for _, body in self.sections:
            found += _LINK.findall(body)
        return tuple(dict.fromkeys(found))

    @property
    def relationships(self) -> tuple[tuple[str, str], ...]:
        """`("depends on", "Ember")` for each well-formed Relationships bullet."""
        for heading, body in self.sections:
            if heading == "Relationships":
                return tuple(
                    (m.group(1), m.group(2))
                    for m in (_RELATIONSHIP.match(line) for line in body.splitlines())
                    if m
                )
        return ()

    def section(self, heading: str) -> str | None:
        for name, body in self.sections:
            if name == heading:
                return body
        return None

    # --- serialisation -------------------------------------------------------

    def render(self) -> str:
        meta = [
            f"title: {_scalar(self.title)}",
            f"page_type: {self.page_type}",
        ]
        if self.entity_type:
            meta.append(f"entity_type: {self.entity_type}")
        meta.append(f"sources: [{', '.join(self.sources)}]")
        meta.append(f"updated: {self.updated.isoformat()}")

        parts = [
            "---\n" + "\n".join(meta) + "\n---",
            f"# {self.title}",
            f"> {self.summary}",
        ]
        parts += [f"## {heading}\n\n{body}" for heading, body in self.sections]
        parts.append(
            f"## {_DERIVED_SECTION}\n\n" + "\n".join(f"- {s}" for s in self.sources)
        )
        return "\n\n".join(parts) + "\n"

    @classmethod
    def parse(cls, text: str) -> WikiPage:
        match = _FRONTMATTER.match(text)
        if not match:
            raise ValueError("page has no frontmatter block; it was not written by the compiler")
        meta = yaml.safe_load(match.group(1)) or {}

        summary = ""
        sections: list[tuple[str, list[str]]] = []
        for line in text[match.end() :].splitlines():
            heading = _HEADING.match(line)
            if heading:
                sections.append((heading.group(1).strip(), []))
            elif sections:
                sections[-1][1].append(line)
            elif line.startswith("> ") and not summary:
                summary = line[2:].strip()

        return cls(
            title=str(meta["title"]),
            page_type=meta["page_type"],
            entity_type=meta.get("entity_type"),
            sources=tuple(meta.get("sources") or ()),
            updated=_as_date(meta["updated"]),
            summary=summary,
            sections=tuple(
                (heading, "\n".join(body).strip())
                for heading, body in sections
                if heading != _DERIVED_SECTION
            ),
        )


def _scalar(value: str) -> str:
    # A title containing a colon would otherwise parse back as a nested mapping.
    return f'"{value}"' if ":" in value or value.startswith(("[", "{", "*", "&")) else value


def _as_date(value: object) -> dt.date:
    return value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value))
