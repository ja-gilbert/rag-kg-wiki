from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator

# The contract every approach returns and the frontend renders. It is listed
# explicitly, and tested against the dataclass, because "same envelope, same
# layout" is what makes the three columns comparable -- a field added for one
# approach has to be added for all three and for the UI in the same change.
ENVELOPE_FIELDS = (
    "approach",  # "rag" | "kg" | "wiki"
    "label",  # display name
    "answer",  # the generated text
    "evidence",  # chunks, paths, or pages
    "citations",  # source ids
    "trace",  # what it actually did, step by step, in plain English
    "detail",  # approach-specific payload (plans, paths, scores, highlights)
    "ms",  # per-phase timing plus total
    "tokens_est",  # evidence size handed to the generator
    "confident",
    "note",  # the honest caveat when confident is False
)


@dataclass(frozen=True)
class Answer:
    approach: str
    label: str
    answer: str
    evidence: list[dict[str, Any]]
    citations: list[str]
    trace: list[str]
    detail: dict[str, Any]
    ms: dict[str, float]
    tokens_est: int
    confident: bool
    note: str | None = None

    def __post_init__(self) -> None:
        # Non-negotiable #7: admitting ignorance is a feature, but an
        # unexplained shrug looks exactly like a broken approach. Making this
        # structural means no approach can ship one by accident.
        if not self.confident and not self.note:
            raise ValueError(
                f"{self.approach}: confident=False requires a note explaining why"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Approach:
    """Base class for the three answering strategies.

    Subclasses set `name` and `label` and implement `answer`. Everything else
    they share -- the envelope, the timer, the token estimate -- lives here so
    the three stay comparable by construction rather than by discipline.
    """

    name = "base"
    label = "Base"

    def answer(self, question: str) -> Answer:
        raise NotImplementedError


class PhaseTimer:
    """Wall-clock per phase, plus a total, in milliseconds.

    The total is measured from construction rather than summed from the
    phases, so time spent between them still shows up somewhere.
    """

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._phases: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            # Recorded in `finally` so a phase that raises still reports its
            # cost -- otherwise a failing approach shows ms={} and the
            # scoreboard quietly credits it with being fast.
            elapsed = (time.perf_counter() - started) * 1000
            self._phases[name] = self._phases.get(name, 0.0) + elapsed

    def ms(self) -> dict[str, float]:
        total = (time.perf_counter() - self._started) * 1000
        return {**{k: round(v, 1) for k, v in self._phases.items()}, "total": round(total, 1)}


def estimate_tokens(texts: Iterable[str]) -> int:
    """Rough evidence size: ~4 characters per token.

    Deliberately not a real tokeniser. The number exists so the scoreboard can
    compare how much text each architecture had to hand its generator, and that
    comparison is unaffected by the constant.
    """
    return sum(len(text) for text in texts) // 4
