from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from approaches.base import Answer, Approach
from approaches.kg import KgApproach
from approaches.rag import RagApproach
from approaches.wiki import WikiApproach
from core.config import DEFAULT_CONFIG_PATH, load_config
from core.llm import build_llm
from scripts.artefacts import ArtefactMissing, load_kg, load_rag, load_wiki


def load_approaches(cfg: dict[str, Any]) -> list[Approach]:
    """Every approach, sharing one generator.

    Non-negotiable #5: they get the same backend and the same prompt, so any
    difference between the answers is a difference in what each one found.
    """
    store, bm25, embedder = load_rag(cfg)
    graph, ontology = load_kg(cfg)
    llm = build_llm(cfg["llm"])
    return [
        RagApproach(store=store, bm25=bm25, embedder=embedder, llm=llm, cfg=cfg["rag"]),
        KgApproach(graph=graph, ontology=ontology, llm=llm, cfg=cfg["kg"]),
        WikiApproach(library=load_wiki(cfg), llm=llm, cfg=cfg["wiki"]),
    ]


def ask(question: str, cfg: dict[str, Any]) -> list[Answer]:
    return [approach.answer(question) for approach in load_approaches(cfg)]


def ask_all(cfg: dict[str, Any]) -> list[tuple[str, list[Answer]]]:
    # Artefacts load once and every question reuses them, which is also what
    # makes the per-question timings comparable to each other.
    approaches = load_approaches(cfg)
    return [
        (row["q"], [approach.answer(row["q"]) for approach in approaches])
        for row in _questions(cfg)
    ]


def _questions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw = Path(cfg["paths"]["questions"]).read_text(encoding="utf-8")
    return yaml.safe_load(raw)["questions"]


def _render(question: str, answers: list[Answer]) -> str:
    lines = [f"Q: {question}", ""]
    for answer in answers:
        verdict = "" if answer.confident else "  [not confident]"
        lines.append(f"  {answer.label}{verdict}  {answer.ms['total']:.0f}ms")
        lines.append(f"    {answer.answer.strip()}")
        if answer.citations:
            lines.append(f"    sources: {', '.join(answer.citations)}")
        if answer.note:
            lines.append(f"    note: {answer.note}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the corpus a question from the terminal.")
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument("--all", action="store_true", help="run every question in questions.yaml")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        if args.all:
            for question, answers in ask_all(cfg):
                print(_render(question, answers))
            return
        if not args.question:
            parser.error("give a question, or --all")
        question = " ".join(args.question)
        print(_render(question, ask(question, cfg)))
    except ArtefactMissing as exc:
        # The one error a first-time reader is most likely to hit, and a stack
        # trace here would send them looking in the wrong place entirely.
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
