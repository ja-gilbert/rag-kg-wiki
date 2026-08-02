from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.registry import Lab, build_lab
from approaches.base import Answer, Approach
from core.config import DEFAULT_CONFIG_PATH, load_config
from scripts.artefacts import BUILD_COMMAND
from wikigen.lint import lint

STATIC = Path(__file__).resolve().parent / "static"


class AskRequest(BaseModel):
    question: str


def create_app(cfg: dict[str, Any] | None = None) -> FastAPI:
    """The whole server, over one `Lab` resolved at startup.

    Taking the config as an argument rather than reading it at import time is
    what lets the tests point a whole app at a temporary build directory.
    """
    lab = build_lab(cfg if cfg is not None else load_config(DEFAULT_CONFIG_PATH))
    api = FastAPI(title="kb-lab", description="One question, three architectures.")
    api.state.lab = lab

    # --- the comparison ------------------------------------------------------

    @api.post("/api/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        ready = _ready(api.state.lab)
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="ask something")

        answers = [_answer(approach, question) for approach in ready.approaches]
        return {
            "question": question,
            "answers": [a.to_dict() for a in answers],
            "scoreboard": scoreboard(answers),
        }

    @api.get("/api/questions")
    def questions() -> dict[str, Any]:
        return {"questions": [dict(row) for row in api.state.lab.questions]}

    # --- the tabs ------------------------------------------------------------

    @api.get("/api/graph")
    def graph() -> dict[str, Any]:
        return _ready(api.state.lab).graph.to_dict()

    @api.get("/api/wiki")
    def wiki() -> dict[str, Any]:
        library = _ready(api.state.lab).library
        return {
            "pages": [
                {"title": e.title, "summary": e.summary, "category": e.category}
                for e in library.entries
            ]
        }

    @api.get("/api/wiki/{title}")
    def wiki_page(title: str) -> dict[str, Any]:
        library = _ready(api.state.lab).library
        if not library.has(title):
            raise HTTPException(status_code=404, detail=f"no wiki page titled {title!r}")
        page = library.page(title)
        return {
            "title": page.title,
            "page_type": page.page_type,
            "entity_type": page.entity_type,
            "summary": page.summary,
            "updated": page.updated.isoformat(),
            "sections": [{"heading": h, "body": b} for h, b in page.sections],
            "sources": list(page.sources),
            "links": list(page.links),
            "backlinks": list(library.backlinks(title)),
            "markdown": page.render(),
        }

    @api.get("/api/lint")
    def lint_report() -> dict[str, Any]:
        # Deliberately not `wikigen.lint.lint_wiki`: that one appends a LINT
        # entry to log.md, and a button someone can hold down should not be
        # able to write to an append-only journal a thousand times.
        ready = _ready(api.state.lab)
        report = lint(
            ready.library,
            ready.documents,
            ready.ontology,
            stale_after_days=ready.cfg["wiki"]["stale_after_days"],
        )
        return {
            "page_count": report.page_count,
            "seconds": round(report.seconds, 4),
            "ok": report.ok,
            "findings": [
                {
                    "check": f.check,
                    "severity": f.severity,
                    "subject": f.subject,
                    "detail": f.detail,
                }
                for f in report.findings
            ],
        }

    @api.get("/api/sources")
    def sources() -> dict[str, Any]:
        return {
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "doc_type": d.doc_type,
                    "date": d.date.isoformat() if d.date else None,
                }
                for d in api.state.lab.documents
            ]
        }

    @api.get("/api/sources/{doc_id}")
    def source(doc_id: str) -> dict[str, Any]:
        document = api.state.lab.document(doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail=f"no document {doc_id!r}")
        return {
            "doc_id": document.doc_id,
            "title": document.title,
            "doc_type": document.doc_type,
            "date": document.date.isoformat() if document.date else None,
            "text": document.body,
        }

    # --- the shell -----------------------------------------------------------

    @api.get("/api/status")
    def status() -> dict[str, Any]:
        current: Lab = api.state.lab
        return {
            "ready": current.ready,
            "problem": current.problem,
            "build_command": BUILD_COMMAND,
            "documents": len(current.documents),
            "questions": len(current.questions),
            "approaches": [
                {"name": a.name, "label": a.label} for a in current.approaches
            ],
            "pages": len(current.library.titles) if current.library else 0,
            "nodes": len(current.graph.nodes) if current.graph else 0,
            "edges": len(current.graph.edges) if current.graph else 0,
            # The knobs the three approaches disagree about, so the UI can say
            # which configuration produced what is on screen.
            "config": {
                "embedding": current.cfg["embedding"]["backend"],
                "chunking": current.cfg["chunking"]["strategy"],
                "llm": current.cfg["llm"]["backend"],
                "rag_mode": current.cfg["rag"]["mode"],
                "rag_top_k": current.cfg["rag"]["top_k"],
                "kg_max_hops": current.cfg["kg"]["max_hops"],
                "wiki_max_pages": current.cfg["wiki"]["max_pages"],
            },
        }

    @api.get("/", response_class=FileResponse)
    def index():
        page = STATIC / "index.html"
        if not page.exists():
            return PlainTextResponse(
                "kb-lab is running. The API is up at /api/status; the frontend "
                "has not been built into app/static yet.\n"
            )
        return FileResponse(page)

    if STATIC.exists():
        api.mount("/static", StaticFiles(directory=STATIC), name="static")

    return api


def scoreboard(answers: list[Answer]) -> list[dict[str, Any]]:
    """The comparison rows, computed in one place for all three columns.

    `derivation_shown` keys off a `path` in the detail payload rather than off
    the approach's name: the row is about what the answer can show you, and a
    server that special-cased "kg" would be asserting the result it wants
    rather than reading it. The wiki's provenance is real but is a chain of
    citations, not a derivation of the answer, so it reads false here.
    """
    return [
        {
            "approach": a.approach,
            "label": a.label,
            "latency_ms": a.ms.get("total", 0.0),
            "evidence_tokens": a.tokens_est,
            "documents_touched": len(set(a.citations)),
            "derivation_shown": bool(a.detail.get("path")),
            "admitted_ignorance": not a.confident,
        }
        for a in answers
    ]


def _ready(lab: Lab) -> Lab:
    if not lab.ready:
        raise HTTPException(status_code=503, detail=lab.problem)
    return lab


def _answer(approach: Approach, question: str) -> Answer:
    """One approach's answer, or an envelope explaining why there isn't one.

    Sequential rather than threaded, and per-approach rather than per-request:
    with the extractive backend all three are CPU-bound Python, so running them
    together would buy no wall-clock under the GIL while inflating each other's
    latency -- which is the number the scoreboard is built to compare. Catching
    here is what stops one broken approach from blanking the other two columns.
    """
    try:
        return approach.answer(question)
    except Exception as exc:  # noqa: BLE001 -- a demo must not lose two columns to one bug
        return Answer(
            approach=approach.name,
            label=approach.label,
            answer="",
            evidence=[],
            citations=[],
            trace=[f"{approach.name} raised {type(exc).__name__}"],
            detail={"error": f"{type(exc).__name__}: {exc}"},
            ms={"total": 0.0},
            tokens_est=0,
            confident=False,
            note=f"This approach failed: {type(exc).__name__}: {exc}",
        )


def main() -> None:
    # `python -m app.main`, matching how every other entry point in this repo
    # is run. No module-level app: constructing one at import time would make
    # importing this module load every artefact, including from the tests.
    import uvicorn

    cfg = load_config(DEFAULT_CONFIG_PATH)
    server = cfg["server"]
    if server.get("reload"):
        uvicorn.run(
            "app.main:create_app",
            factory=True,
            host=server["host"],
            port=server["port"],
            reload=True,
        )
        return
    uvicorn.run(create_app(cfg), host=server["host"], port=server["port"])


if __name__ == "__main__":
    main()
