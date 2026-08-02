from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from core.chunking import chunk_documents
from core.config import DEFAULT_CONFIG_PATH, load_config
from core.corpus import load_corpus
from core.embeddings import build_embedder
from core.vectorstore import VectorStore
from scripts.artefacts import BuildStats, index_dir


def build_index(cfg: dict[str, Any]) -> BuildStats:
    started = time.perf_counter()
    documents = load_corpus(Path(cfg["paths"]["raw"]))
    chunks = chunk_documents(documents, cfg["chunking"])
    texts = [chunk.text for chunk in chunks]

    # Timed apart from the encoding because it dominates the total and says
    # nothing about the architecture: loading 90MB of weights off disk is a
    # one-time cost that a comparison of build cost should not silently bill
    # to RAG.
    loading = time.perf_counter()
    embedder = build_embedder(cfg["embedding"], texts)
    load_seconds = time.perf_counter() - loading

    VectorStore(chunks, embedder.encode(texts)).save(index_dir(cfg))

    return BuildStats(
        approach="rag",
        artefact=str(index_dir(cfg)),
        count=len(chunks),
        seconds=time.perf_counter() - started,
        note=(
            f"{len(documents)} documents, {cfg['chunking']['strategy']} chunking, "
            f"{cfg['embedding']['backend']} embeddings "
            f"({load_seconds:.1f}s of it loading the model)"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed the corpus into a vector index.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    stats = build_index(load_config(args.config))
    print(f"{stats.count} chunks -> {stats.artefact} in {stats.seconds:.1f}s ({stats.note})")


if __name__ == "__main__":
    main()
