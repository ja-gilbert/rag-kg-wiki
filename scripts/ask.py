from __future__ import annotations

import argparse
import sys
from typing import Any

from app.registry import build_lab
from approaches.base import Answer, Approach
from core.config import DEFAULT_CONFIG_PATH, load_config
from scripts.artefacts import ArtefactMissing


def load_approaches(cfg: dict[str, Any]) -> list[Approach]:
    """Every approach, sharing one generator.

    Wired in `app.registry` and borrowed here rather than built twice: the
    terminal and the browser have to be running the same three objects, or
    non-negotiable #5 holds only by coincidence. The server turns a missing
    artefact into a banner; a CLI has nowhere to put a banner, so it gets the
    exception back.
    """
    lab = build_lab(cfg)
    if not lab.ready:
        raise ArtefactMissing(lab.problem)
    return list(lab.approaches)


def ask(question: str, cfg: dict[str, Any]) -> list[Answer]:
    return [approach.answer(question) for approach in load_approaches(cfg)]


def ask_all(cfg: dict[str, Any]) -> list[tuple[str, list[Answer]]]:
    # Artefacts load once and every question reuses them, which is also what
    # makes the per-question timings comparable to each other.
    lab = build_lab(cfg)
    if not lab.ready:
        raise ArtefactMissing(lab.problem)
    return [
        (row["q"], [approach.answer(row["q"]) for approach in lab.approaches])
        for row in lab.questions
    ]


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
