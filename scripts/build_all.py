from __future__ import annotations

import argparse
from typing import Any

from core.config import DEFAULT_CONFIG_PATH, load_config
from scripts.artefacts import BuildStats
from scripts.build_graph import build_graph
from scripts.build_index import build_index
from scripts.build_wiki import build_wiki

# The asymmetry between these three is the point of the table: RAG embeds in
# seconds, the graph extracts in less, and the wiki does its synthesis once at
# build time instead of on every query the way RAG does.
_BUILDERS = (build_index, build_graph, build_wiki)

_HEADINGS = ("approach", "artefact", "items", "seconds", "notes")


def build_all(cfg: dict[str, Any]) -> list[BuildStats]:
    return [build(cfg) for build in _BUILDERS]


def summary_table(rows: list[BuildStats]) -> str:
    cells = [_HEADINGS] + [
        (r.approach, r.artefact, str(r.count), f"{r.seconds:.1f}", r.note) for r in rows
    ]
    widths = [max(len(row[n]) for row in cells) for n in range(len(_HEADINGS))]
    lines = [
        "  ".join(cell.ljust(widths[n]) for n, cell in enumerate(row)).rstrip()
        for row in cells
    ]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build every artefact the demo needs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    rows = build_all(load_config(args.config))
    print(summary_table(rows))
    print()
    # Worth saying out loud rather than leaving to be inferred from the numbers:
    # the graph's real cost was never the extraction, it was the ontology.
    print(
        "RAG builds from the corpus alone. The graph needs an ontology someone "
        "had to write and tune until no entity was orphaned -- that cost does "
        "not appear in the seconds column."
    )


if __name__ == "__main__":
    main()
