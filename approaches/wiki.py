from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from approaches.base import Answer, Approach, PhaseTimer, estimate_tokens
from core.llm import Evidence
from core.text import stems
from wikigen.library import IndexEntry, WikiLibrary
from wikigen.page import WikiPage

# Navigation, not similarity. This scans index.md the way a person scans a
# table of contents -- titles and one-line summaries only -- then opens whole
# pages and follows their [[links]] one hop.
#
# Non-negotiable #4: nothing here embeds anything. Scoring page *bodies* would
# turn this into RAG with bigger chunks and the comparison would collapse.

_TITLE_WEIGHT = 3.0
_PHRASE_BONUS = 3.0

# A term on more than a quarter of the pages does not tell you which page to
# open. Requiring at least one term below this cut is what lets the wiki say
# "no page covers that" instead of matching every page whose summary happens
# to contain the company name -- which is exactly what the revenue question
# does otherwise.
_COMMON_TERM_SHARE = 0.25


@dataclass(frozen=True)
class Match:
    entry: IndexEntry
    score: float
    hits: tuple[str, ...]


class WikiApproach(Approach):
    name = "wiki"
    label = "LLM wiki"

    def __init__(self, library: WikiLibrary, llm: Any, cfg: dict[str, Any]) -> None:
        self._library = library
        self._llm = llm
        self._max_pages = cfg.get("max_pages", 6)
        self._expansion = cfg.get("backlink_expansion", 1)
        self._document_frequency = _document_frequency(library.entries)
        self._total = max(len(library.entries), 1)

    def answer(self, question: str) -> Answer:
        timer = PhaseTimer()
        with timer.phase("scan"):
            matches = self._scan(question)

        with timer.phase("read"):
            seeds = [self._library.page(m.entry.title) for m in matches[: self._max_pages]]
            followed = self._follow(seeds)
            pages = (seeds + [self._library.page(t) for t in followed])[: self._max_pages]

        with timer.phase("generate"):
            generation = self._llm.generate(question, _evidence(pages)) if pages else None

        blocks = [_page_dict(p) for p in pages]
        return Answer(
            approach=self.name,
            label=self.label,
            answer=generation.text if generation else _NOTHING_FOUND,
            evidence=blocks,
            # Cited twice: the raw documents here, the pages in `detail`. Being
            # able to walk answer -> page -> source document is this approach's
            # real advantage over the other two.
            citations=list(dict.fromkeys(d for b in blocks for d in b["doc_ids"])),
            trace=self._trace(matches, followed, pages),
            detail={
                "matched": [
                    {
                        "title": m.entry.title,
                        "category": m.entry.category,
                        "page_type": "topic" if m.entry.category == "Topics" else "entity",
                        "summary": m.entry.summary,
                        "score": round(m.score, 2),
                        "matched_on": list(m.hits),
                    }
                    for m in matches
                ],
                "followed": list(followed),
                "pages_read": [p.title for p in pages],
                "max_pages": self._max_pages,
            },
            ms=timer.ms(),
            tokens_est=estimate_tokens(b["text"] for b in blocks),
            confident=bool(pages),
            note=None if pages else _NOTHING_FOUND,
        )

    # --- the catalogue scan ---------------------------------------------------

    def _scan(self, question: str) -> list[Match]:
        asked = stems(question)
        lowered = question.lower()
        matches = []
        for entry in self._library.entries:
            if not self._library.has(entry.title):
                continue  # catalogued but unwritten: lint's problem, not ours
            title_hits = asked & stems(entry.title)
            summary_hits = asked & stems(entry.summary)
            if not self._distinctive(title_hits | summary_hits):
                continue

            score = _TITLE_WEIGHT * self._weight(title_hits) + self._weight(summary_hits)
            if entry.title.lower() in lowered:
                score += _PHRASE_BONUS
            matches.append(
                Match(entry, score, tuple(sorted(title_hits | summary_hits)))
            )
        # Coverage first, weight second. A question naming five services is
        # asking how they relate, and no single service page answers that --
        # the page covering three of its terms beats the one matching a rare
        # term very strongly. Without this, "how do they fit together" is
        # answered from five entity pages, which is what RAG already does.
        return sorted(matches, key=lambda m: (-len(m.hits), -m.score, m.entry.title))

    def _distinctive(self, hits: set[str]) -> bool:
        cut = max(1.0, self._total * _COMMON_TERM_SHARE)
        return any(self._document_frequency.get(term, 0) <= cut for term in hits)

    def _weight(self, hits: set[str]) -> float:
        return sum(
            math.log(1 + self._total / self._document_frequency.get(term, self._total))
            for term in hits
        )

    def _follow(self, seeds: list[WikiPage]) -> list[str]:
        """One hop out, in the order the pages link to things."""
        if self._expansion <= 0:
            return []
        seen = {page.title for page in seeds}
        found: list[str] = []
        frontier = list(seeds)
        for _ in range(self._expansion):
            nxt = []
            for page in frontier:
                for target in page.links:
                    if target in seen or not self._library.has(target):
                        continue
                    seen.add(target)
                    found.append(target)
                    nxt.append(self._library.page(target))
            frontier = nxt
        return found

    def _trace(self, matches: list[Match], followed: list[str], pages: list[WikiPage]) -> list[str]:
        steps = [
            f"Scanned index.md -- {self._total} page titles and summaries, no page bodies.",
            (
                f"{len(matches)} page(s) matched on a distinctive term."
                if matches
                else "No page matched on any term specific enough to open it."
            ),
        ]
        if matches:
            steps.append(
                "Best match: "
                f"{matches[0].entry.title} (on {', '.join(matches[0].hits)})."
            )
            steps.append(
                f"Followed {len(followed)} link(s) one hop out from the matched pages."
                if followed
                else "Followed no links: the matched pages already filled the budget."
            )
            steps.append(f"Read {len(pages)} whole page(s) as evidence.")
        return steps


_NOTHING_FOUND = (
    "No page in this wiki covers that. The index has nothing whose title or "
    "summary is specific to the question."
)


def _document_frequency(entries: tuple[IndexEntry, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        for term in stems(entry.title) | stems(entry.summary):
            counts[term] = counts.get(term, 0) + 1
    return counts


def _page_dict(page: WikiPage) -> dict[str, Any]:
    return {
        "title": page.title,
        "page_type": page.page_type,
        "summary": page.summary,
        "doc_ids": list(page.sources),
        # Whole pages, not fragments. The page is the unit the compiler wrote
        # and the unit a reader would open -- but its prose, not its
        # frontmatter: see WikiPage.body().
        "text": page.body(),
    }


def _evidence(pages: list[WikiPage]) -> list[Evidence]:
    return [Evidence(citation=page.title, text=page.body()) for page in pages]
