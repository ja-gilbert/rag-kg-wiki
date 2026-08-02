from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from core.corpus import Document, load_corpus
from kgraph.ontology import Ontology
from wikigen import journal
from wikigen.library import WikiLibrary
from wikigen.page import WikiPage

# The third operation, and the one neither other approach has any answer to: a
# vector index cannot notice that two of its chunks contradict each other, and
# it certainly cannot notice that one of them has been superseded.
#
# Lint reads the compiled wiki and the raw corpus it was compiled from. It never
# reads the knowledge graph -- a contradiction that never reached a page is not
# a problem with the wiki, and a check that consulted the graph would be
# reporting on the extraction layer while claiming to report on the pages.

# Predicates that admit exactly one object per subject. Two of them and the
# wiki is asserting something that cannot be true.
FUNCTIONAL_PREDICATES = ("owned_by", "led_by", "caused_by", "supersedes")

CHECKS = (
    "orphan_pages",
    "broken_links",
    "contradictions",
    "stale_pages",
    "superseded_refs",
    "coverage_gaps",
    "index_drift",
)

# Errors are the findings that make the wiki wrong or unnavigable, and they are
# what the exit code keys off so this can run in CI. Warnings are the findings
# that make it incomplete or ageing -- worth surfacing, not worth failing a
# build over, because a corpus that has not been touched in six months is a fact
# about the corpus rather than a defect in the compiler.
SEVERITY = {
    "broken_links": "error",
    "contradictions": "error",
    "superseded_refs": "error",
    "index_drift": "error",
    "orphan_pages": "warning",
    "stale_pages": "warning",
    "coverage_gaps": "warning",
}


@dataclass(frozen=True)
class Finding:
    check: str
    severity: str
    subject: str  # the page title, or the doc id, the finding is about
    detail: str

    def render(self) -> str:
        return f"{self.subject}: {self.detail}"


@dataclass(frozen=True)
class LintReport:
    findings: tuple[Finding, ...]
    page_count: int
    seconds: float

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def of(self, check: str) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.check == check)

    def render(self) -> str:
        lines = [
            f"LINT {self.page_count} pages: {_plural(len(self.errors), 'error')}, "
            f"{_plural(len(self.warnings), 'warning')} ({self.seconds:.2f}s)"
        ]
        for check in CHECKS:
            found = self.of(check)
            if not found:
                continue
            # ASCII only: this goes to a Windows console, which is not
            # necessarily UTF-8, and a CI runner should not die on a dash.
            lines.append(f"\n  {check} ({SEVERITY[check]}): {len(found)}")
            lines += [f"    - {f.render()}" for f in found]
        return "\n".join(lines)


def lint_wiki(cfg: dict[str, Any]) -> LintReport:
    root = Path(cfg["paths"]["wiki"])
    report = lint(
        WikiLibrary.load(root),
        load_corpus(Path(cfg["paths"]["raw"])),
        Ontology.load(cfg["paths"]["ontology"]),
        stale_after_days=cfg["wiki"]["stale_after_days"],
    )
    journal.append(
        root,
        f"LINT {report.page_count} pages: {_plural(len(report.errors), 'error')}, "
        f"{_plural(len(report.warnings), 'warning')}",
    )
    return report


def lint(
    library: WikiLibrary,
    documents: Sequence[Document],
    ontology: Ontology,
    stale_after_days: int,
    as_of: dt.date | None = None,
) -> LintReport:
    started = time.perf_counter()
    as_of = as_of or _corpus_date(documents)
    findings = (
        _orphan_pages(library)
        + _broken_links(library)
        + _contradictions(library, ontology)
        + _stale_pages(library, documents, stale_after_days, as_of)
        + _superseded_refs(library, ontology)
        + _coverage_gaps(library, documents)
        + _index_drift(library)
    )
    return LintReport(tuple(findings), len(library.titles), time.perf_counter() - started)


# --- the checks --------------------------------------------------------------


def _orphan_pages(library: WikiLibrary) -> list[Finding]:
    """Entity pages nothing links to.

    Topic pages are excluded deliberately. They are the wiki's entry points --
    catalogued in `index.md`, linking outward to everything they synthesise, and
    linked back to by nothing. Reporting all four on every run would bury the
    case this check exists for: an entity page the wiki has lost track of.
    """
    return [
        _finding("orphan_pages", title, "no other page links here")
        for title in library.titles
        if library.page(title).page_type != "topic" and not library.backlinks(title)
    ]


def _broken_links(library: WikiLibrary) -> list[Finding]:
    return [
        _finding("broken_links", title, f"[[{target}]] resolves to no page")
        for title in library.titles
        for target in library.page(title).links
        if not library.has(target)
    ]


def _contradictions(library: WikiLibrary, ontology: Ontology) -> list[Finding]:
    """Two objects where a functional predicate allows one.

    Only the forward label counts. `owned by` is one-to-one, but its inverse
    `owns` is one-to-many -- a team owning three services is the normal case,
    and flagging it would bury every real contradiction.
    """
    forward = {
        ontology.relations[predicate].label: predicate
        for predicate in FUNCTIONAL_PREDICATES
        if predicate in ontology.relations
    }

    findings = []
    for title in library.titles:
        grouped: dict[str, list[str]] = {}
        for label, other in library.page(title).relationships:
            if label in forward and other not in grouped.setdefault(label, []):
                grouped[label].append(other)
        for label, others in grouped.items():
            if len(others) > 1:
                stated = " and ".join(f"[[{other}]]" for other in others)
                findings.append(
                    _finding(
                        "contradictions",
                        title,
                        f"{forward[label]} takes one object, but this page says "
                        f"{label} {stated}",
                    )
                )
    return findings


def _stale_pages(
    library: WikiLibrary, documents: Sequence[Document], stale_after_days: int, as_of: dt.date
) -> list[Finding]:
    """Pages whose newest source predates the corpus by more than the window.

    Measured against the corpus's own most recent document rather than against
    today. Staleness here means "this page has been left behind while the rest
    of the wiki moved on", which is the thing worth reporting; measuring against
    the wall clock would instead mark every page in a fixed corpus stale as soon
    as the corpus stopped being updated, and mark a different set every day.
    """
    dated = {d.doc_id: d.date for d in documents if d.date}
    cutoff = as_of - dt.timedelta(days=stale_after_days)

    findings = []
    for title in library.titles:
        newest = max(
            (dated[s] for s in library.page(title).sources if s in dated), default=None
        )
        if newest and newest < cutoff:
            findings.append(
                _finding(
                    "stale_pages",
                    title,
                    f"newest source is {newest}, {(as_of - newest).days} days behind "
                    f"the corpus ({as_of})",
                )
            )
    return findings


def _superseded_refs(library: WikiLibrary, ontology: Ontology) -> list[Finding]:
    """Pages that link to a superseded page without passing that on.

    The most dangerous failure any knowledge base has: answering fluently and
    with a citation from a document that has been replaced. Which pages are
    superseded is read off the wiki's own relationship bullets, which the
    compiler wrote from prose -- nothing here is tagged by hand.
    """
    relation = ontology.relations.get("supersedes")
    if relation is None:
        return []

    superseded = set()
    for title in library.titles:
        for label, other in library.page(title).relationships:
            if label == relation.label:
                superseded.add(other)
            elif label == relation.inverse_label:
                superseded.add(title)

    findings = []
    for title in library.titles:
        page = library.page(title)
        text = _text(page).lower()
        if relation.label in text or relation.inverse_label in text:
            continue  # the page says so somewhere, which is all this check wants
        findings += [
            _finding(
                "superseded_refs",
                title,
                f"links to [[{target}]], which has been superseded, without saying so",
            )
            for target in page.links
            if target in superseded
        ]
    return findings


def _coverage_gaps(library: WikiLibrary, documents: Sequence[Document]) -> list[Finding]:
    cited = {source for title in library.titles for source in library.page(title).sources}
    return [
        _finding("coverage_gaps", d.doc_id, "no page cites this document")
        for d in documents
        if d.doc_id not in cited
    ]


def _index_drift(library: WikiLibrary) -> list[Finding]:
    """`index.md` and `wiki/pages/` disagreeing.

    An error rather than a warning because the query layer scans the index and
    nothing else: a page missing from it is a page the wiki cannot reach, however
    well written it is.
    """
    catalogued = {entry.title: entry for entry in library.entries}

    findings = []
    for title, entry in catalogued.items():
        if not library.has(title):
            findings.append(
                _finding("index_drift", title, "catalogued in index.md but no page was written")
            )
        elif library.page(title).summary != entry.summary:
            findings.append(
                _finding("index_drift", title, "index.md quotes a summary the page no longer carries")
            )
    findings += [
        _finding("index_drift", title, "page exists but index.md does not list it")
        for title in library.titles
        if title not in catalogued
    ]
    return findings


# --- internals ---------------------------------------------------------------


def _finding(check: str, subject: str, detail: str) -> Finding:
    # Severity lives in one table rather than at each call site, so the exit
    # code cannot drift from what the checks claim to be.
    return Finding(check=check, severity=SEVERITY[check], subject=subject, detail=detail)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _text(page: WikiPage) -> str:
    return "\n".join([page.summary, *(f"{h}\n{b}" for h, b in page.sections)])


def _corpus_date(documents: Sequence[Document]) -> dt.date:
    return max((d.date for d in documents if d.date), default=dt.date.today())
