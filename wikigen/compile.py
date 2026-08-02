from __future__ import annotations

import dataclasses
import datetime as dt
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.corpus import load_corpus
from kgraph.extract import extract_triples
from kgraph.graph import KnowledgeGraph
from kgraph.ontology import Gazetteer, Ontology
from wikigen import journal
from wikigen.entity_pages import entity_page
from wikigen.page import WikiPage
from wikigen.sources import index_sources
from wikigen.topic_pages import TOPIC_DOC_TYPES, topic_pages

# ingest -> compile. The raw corpus is read and never written; everything under
# wiki/ is generated and owned entirely by this module. Compilation is
# deterministic: same corpus in, byte-identical pages out, which is what makes
# a diff between two compiles worth reading.

_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class CompileResult:
    written: tuple[str, ...]
    page_count: int
    doc_count: int
    seconds: float


def compile_wiki(cfg: dict[str, Any]) -> CompileResult:
    started = time.perf_counter()
    today = dt.date.today()

    documents = load_corpus(Path(cfg["paths"]["raw"]))
    ontology = Ontology.load(cfg["paths"]["ontology"])
    gazetteer = Gazetteer(ontology)
    graph = KnowledgeGraph(extract_triples(documents, ontology), ontology)
    sources = index_sources(documents, ontology, gazetteer)

    pages = [
        entity_page(entity, sources, graph, gazetteer, today)
        for entity in sorted(ontology.entities, key=lambda e: e.name)
        if entity.name in sources.primary
    ]
    pages += topic_pages(graph, sources, ontology, today)
    pages = _cover_every_document(pages, documents)

    root = Path(cfg["paths"]["wiki"])
    written = _write(root, pages)
    result = CompileResult(
        written=tuple(str(p) for p in written),
        page_count=len(pages),
        doc_count=len(documents),
        seconds=time.perf_counter() - started,
    )
    journal.append(root, f"COMPILE {result.page_count} pages from {result.doc_count} documents")
    return result


def _cover_every_document(pages: list[WikiPage], documents: list) -> list[WikiPage]:
    """Attach any uncited document to the topic page that answers for its type.

    A document nothing cites is invisible to the wiki however good the pages
    are, so the gap is closed at compile time rather than left for lint to
    report. Which topic gets it is decided by document type, not by guessing.
    """
    cited = {source for page in pages for source in page.sources}
    orphaned = [d for d in documents if d.doc_id not in cited]
    if not orphaned:
        return pages

    by_title = {page.title: page for page in pages}
    for document in orphaned:
        for title, doc_types in TOPIC_DOC_TYPES.items():
            if document.doc_type in doc_types and title in by_title:
                page = by_title[title]
                by_title[title] = dataclasses.replace(
                    page, sources=tuple(sorted({*page.sources, document.doc_id}))
                )
                break
    return [by_title.get(page.title, page) for page in pages]


def _write(root: Path, pages: list[WikiPage]) -> list[Path]:
    pages_dir = root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for page in pages:
        path = pages_dir / f"{slug(page.title)}.md"
        path.write_text(page.render(), encoding="utf-8")
        written.append(path)

    index = root / "index.md"
    index.write_text(_index(pages), encoding="utf-8")
    written.append(index)
    return written


def _index(pages: list[WikiPage]) -> str:
    """The catalogue the query layer scans instead of embedding anything.

    Titles and one-line summaries only. That constraint is the whole design:
    scoring against page bodies would make this vector search with extra steps.
    """
    lines = [
        "# Index",
        "",
        "Every page in this wiki, grouped by kind. The summaries are the ones the",
        "pages carry; the query layer scores questions against this file alone.",
        "",
    ]
    entity_types = dict.fromkeys(
        page.entity_type for page in pages if page.page_type == "entity"
    )
    for entity_type in entity_types:
        lines.append(f"## {entity_type}")
        lines.append("")
        lines += [
            f"- [[{page.title}]] — {page.summary}"
            for page in pages
            if page.entity_type == entity_type
        ]
        lines.append("")

    lines.append("## Topics")
    lines.append("")
    lines += [
        f"- [[{page.title}]] — {page.summary}" for page in pages if page.page_type == "topic"
    ]
    return "\n".join(lines) + "\n"


def slug(title: str) -> str:
    return _SLUG.sub("-", title.lower()).strip("-")
