from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from wikigen.page import WikiPage

# Read side of the compiler. The query layer talks to this and never touches
# the filesystem itself, which is also what keeps "scan the index, not the
# bodies" enforceable in one place.

_ENTRY = re.compile(r"^- \[\[([^\[\]]+)\]\](?:\s+—\s+(.*))?$")
_CATEGORY = re.compile(r"^## (.+)$")


@dataclass(frozen=True)
class IndexEntry:
    """One line of the catalogue: all the query layer is allowed to score on."""

    title: str
    summary: str
    category: str


class WikiLibrary:
    def __init__(self, entries: tuple[IndexEntry, ...], pages: dict[str, WikiPage]) -> None:
        self.entries = entries
        self._pages = pages
        self._backlinks: dict[str, list[str]] = {}
        for page in pages.values():
            for target in page.links:
                self._backlinks.setdefault(target, []).append(page.title)

    @classmethod
    def load(cls, root: Path | str) -> WikiLibrary:
        directory = Path(root)
        index = directory / "index.md"
        if not index.exists():
            raise FileNotFoundError(
                f"no wiki index at {index}. Run `python -m scripts.build_all` to "
                "compile the wiki first."
            )
        pages = {}
        for path in sorted((directory / "pages").glob("*.md")):
            page = WikiPage.parse(path.read_text(encoding="utf-8"))
            pages[page.title] = page
        return cls(_parse_index(index.read_text(encoding="utf-8")), pages)

    def page(self, title: str) -> WikiPage:
        return self._pages[title]

    def has(self, title: str) -> bool:
        return title in self._pages

    def backlinks(self, title: str) -> tuple[str, ...]:
        return tuple(sorted(self._backlinks.get(title, ())))


def _parse_index(text: str) -> tuple[IndexEntry, ...]:
    entries: list[IndexEntry] = []
    category = ""
    for line in text.splitlines():
        heading = _CATEGORY.match(line)
        if heading:
            category = heading.group(1).strip()
            continue
        entry = _ENTRY.match(line)
        if entry:
            entries.append(
                IndexEntry(
                    title=entry.group(1),
                    summary=(entry.group(2) or "").strip(),
                    category=category,
                )
            )
    return tuple(entries)
