from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from core.config import DEFAULT_CONFIG_PATH, load_config
from core.corpus import load_corpus
from kgraph.extract import extract_triples, extraction_stats
from kgraph.ontology import Ontology
from scripts.artefacts import BuildStats, build_dir, triples_path


def build_graph(cfg: dict[str, Any]) -> BuildStats:
    started = time.perf_counter()
    ontology = Ontology.load(cfg["paths"]["ontology"])
    triples = extract_triples(load_corpus(Path(cfg["paths"]["raw"])), ontology)

    build_dir(cfg).mkdir(parents=True, exist_ok=True)
    # Written as triples rather than as a serialised graph: the graph rebuilds
    # from them in milliseconds, and a flat list with a sentence per row is
    # something a human can actually audit.
    triples_path(cfg).write_text(
        json.dumps([dataclasses.asdict(t) for t in triples], indent=2),
        encoding="utf-8",
    )

    stats = extraction_stats(triples, ontology)
    orphans = ", ".join(stats.orphans) if stats.orphans else "none"
    return BuildStats(
        approach="kg",
        artefact=str(triples_path(cfg)),
        count=stats.triple_count,
        seconds=time.perf_counter() - started,
        note=(
            f"{len(stats.entities_linked)}/{stats.entities_declared} entities linked, "
            f"orphans: {orphans}"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract typed relations into a graph.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    stats = build_graph(load_config(args.config))
    print(f"{stats.count} triples -> {stats.artefact} in {stats.seconds:.1f}s ({stats.note})")


if __name__ == "__main__":
    main()
