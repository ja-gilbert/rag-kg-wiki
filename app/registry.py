from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from approaches.base import Approach
from approaches.kg import KgApproach
from approaches.rag import RagApproach
from approaches.wiki import WikiApproach
from core.corpus import Document, load_corpus
from core.llm import build_llm
from kgraph.graph import KnowledgeGraph
from kgraph.ontology import Ontology
from scripts.artefacts import ArtefactMissing, load_kg, load_rag, load_wiki
from wikigen.library import WikiLibrary

# config -> artefacts -> approaches, resolved once. Nothing here imports the web
# framework: the CLI builds the same object over the same wiring, so there is
# exactly one place where the three approaches are constructed and no way for
# the terminal and the browser to be running different things.


@dataclass(frozen=True)
class Lab:
    """Everything the server answers from.

    A degraded Lab -- artefacts not built yet -- still carries the corpus and
    the demo questions, because those are read straight from `data/` and never
    generated. So the sources tab and the question list work even while the
    banner is telling you to run the build.
    """

    cfg: dict[str, Any]
    documents: tuple[Document, ...]
    questions: tuple[dict[str, Any], ...]
    approaches: tuple[Approach, ...] = ()
    graph: KnowledgeGraph | None = None
    ontology: Ontology | None = None
    library: WikiLibrary | None = None
    problem: str | None = None

    @property
    def ready(self) -> bool:
        return self.problem is None

    def document(self, doc_id: str) -> Document | None:
        return next((d for d in self.documents if d.doc_id == doc_id), None)


def build_lab(cfg: dict[str, Any]) -> Lab:
    documents = tuple(load_corpus(Path(cfg["paths"]["raw"])))
    questions = tuple(_questions(cfg))

    try:
        store, bm25, embedder = load_rag(cfg)
        graph, ontology = load_kg(cfg)
        library = load_wiki(cfg)
    except ArtefactMissing as exc:
        # A demo that dies before it renders is a demo nobody sees. The missing
        # artefact becomes a banner, not an import-time traceback.
        return Lab(cfg=cfg, documents=documents, questions=questions, problem=str(exc))

    # Non-negotiable #5: one generator, shared. Any difference on screen has to
    # be a difference in what the architecture found.
    llm = build_llm(cfg["llm"])
    return Lab(
        cfg=cfg,
        documents=documents,
        questions=questions,
        approaches=(
            RagApproach(store=store, bm25=bm25, embedder=embedder, llm=llm, cfg=cfg["rag"]),
            KgApproach(graph=graph, ontology=ontology, llm=llm, cfg=cfg["kg"]),
            WikiApproach(library=library, llm=llm, cfg=cfg["wiki"]),
        ),
        graph=graph,
        ontology=ontology,
        library=library,
    )


def _questions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw = Path(cfg["paths"]["questions"]).read_text(encoding="utf-8")
    return yaml.safe_load(raw)["questions"]
