from __future__ import annotations

import argparse
from typing import Any

from core.config import DEFAULT_CONFIG_PATH, load_config
from scripts.artefacts import BuildStats
from wikigen.compile import compile_wiki


def build_wiki(cfg: dict[str, Any]) -> BuildStats:
    result = compile_wiki(cfg)
    return BuildStats(
        approach="wiki",
        artefact=cfg["paths"]["wiki"],
        count=result.page_count,
        seconds=result.seconds,
        note=(
            f"compiled from {result.doc_count} documents; synthesis happens here, "
            "once, instead of on every query"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the raw corpus into a wiki.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    stats = build_wiki(load_config(args.config))
    print(f"{stats.count} pages -> {stats.artefact} in {stats.seconds:.1f}s ({stats.note})")


if __name__ == "__main__":
    main()
