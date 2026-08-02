from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.embeddings import Embedder, build_embedder
from core.vectorstore import BM25Index, VectorStore
from kgraph.extract import Triple
from kgraph.graph import KnowledgeGraph
from kgraph.ontology import Ontology
from wikigen.library import WikiLibrary

BUILD_COMMAND = "python -m scripts.build_all"


class ArtefactMissing(FileNotFoundError):
    """A generated artefact is absent.

    Its own type so callers -- the CLI now, the server in step 9 -- can turn it
    into "run the build first" rather than leaking a stack trace about a
    missing .npy at someone who has simply not built the index yet.
    """


@dataclass(frozen=True)
class BuildStats:
    approach: str
    artefact: str
    count: int
    seconds: float
    note: str = ""


def build_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["build"])


def index_dir(cfg: dict[str, Any]) -> Path:
    return build_dir(cfg) / "index"


def triples_path(cfg: dict[str, Any]) -> Path:
    return build_dir(cfg) / "triples.json"


def load_rag(cfg: dict[str, Any]) -> tuple[VectorStore, BM25Index, Embedder]:
    try:
        store = VectorStore.load(index_dir(cfg))
    except FileNotFoundError as exc:
        raise ArtefactMissing(
            f"No vector index at {index_dir(cfg)}. Run `{BUILD_COMMAND}` first."
        ) from exc

    texts = [chunk.text for chunk in store.chunks]
    # The question has to be embedded the same way the chunks were, so the
    # embedder is rebuilt from the same config rather than stored alongside.
    return store, BM25Index(store.chunks), build_embedder(cfg["embedding"], texts)


def load_kg(cfg: dict[str, Any]) -> tuple[KnowledgeGraph, Ontology]:
    path = triples_path(cfg)
    if not path.exists():
        raise ArtefactMissing(
            f"No extracted triples at {path}. Run `{BUILD_COMMAND}` first."
        )
    ontology = Ontology.load(cfg["paths"]["ontology"])
    triples = [Triple(**row) for row in json.loads(path.read_text(encoding="utf-8"))]
    return KnowledgeGraph(triples, ontology), ontology


def load_wiki(cfg: dict[str, Any]) -> WikiLibrary:
    try:
        return WikiLibrary.load(cfg["paths"]["wiki"])
    except FileNotFoundError as exc:
        raise ArtefactMissing(
            f"No compiled wiki at {cfg['paths']['wiki']}. Run `{BUILD_COMMAND}` first."
        ) from exc
