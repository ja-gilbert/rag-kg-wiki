from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

from core.corpus import split_sentences
from core.text import terms

# One backend interface and one prompt for all three approaches (non-negotiable
# #5). Whatever differs on screen has to be a difference in the evidence the
# architecture found, so the generator is deliberately the boring part.

_INSTRUCTION = (
    "Answer the question using only the numbered evidence below.\n"
    "Cite the evidence you use as [1], [2] and so on, directly after the claim "
    "it supports.\n"
    "If the evidence does not contain the answer, say so plainly rather than "
    "guessing."
)

NO_EVIDENCE_TEXT = "No evidence was retrieved for this question."


@dataclass(frozen=True)
class Evidence:
    """One numbered block in the prompt: some text and the id it came from."""

    citation: str
    text: str


@dataclass(frozen=True)
class Generation:
    text: str
    backend: str  # who actually produced the text, not who was asked to
    fell_back: bool = False
    note: str | None = None


def build_prompt(question: str, evidence: Sequence[Evidence]) -> str:
    blocks = "\n".join(
        f"[{n}] ({item.citation}) {item.text}" for n, item in enumerate(evidence, 1)
    )
    return f"{_INSTRUCTION}\n\nEvidence:\n{blocks}\n\nQuestion: {question}\n\nAnswer:"


# --------------------------------------------------------------------------
# extractive
# --------------------------------------------------------------------------

class ExtractiveBackend:
    """Ranks evidence sentences against the question and stitches them together.

    No model, no key, no network. It cannot hallucinate and it cannot
    paraphrase, because it only ever reorders sentences that already exist --
    and watching a demo pay both halves of that trade is the point of shipping
    it as the default.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 6) -> None:
        self._max_sentences = max_sentences

    def prompt_for(self, question: str, evidence: Sequence[Evidence]) -> str:
        return build_prompt(question, evidence)

    def generate(self, question: str, evidence: Sequence[Evidence]) -> Generation:
        return Generation(text=self._stitch(question, evidence), backend=self.name)

    def _stitch(self, question: str, evidence: Sequence[Evidence]) -> str:
        if not evidence:
            return NO_EVIDENCE_TEXT

        asked = terms(question)
        # Position is kept so the survivors can be re-sorted into reading order:
        # ranking decides what is said, the retriever decides in what order.
        scored: list[tuple[int, int, str, int]] = []
        for number, item in enumerate(evidence, 1):
            for position, sentence in enumerate(split_sentences(item.text)):
                overlap = len(asked & terms(sentence))
                scored.append((overlap, number, sentence, position))

        # Sort by score, then by the order the evidence arrived in, so ties
        # resolve the same way every run -- a demo that reshuffles its own
        # answer between identical questions is worse than useless.
        ranked = sorted(scored, key=lambda s: (-s[0], s[1], s[3]))
        kept = sorted(ranked[: self._max_sentences], key=lambda s: (s[1], s[3]))
        if not kept:
            return NO_EVIDENCE_TEXT
        return " ".join(f"{sentence} [{number}]" for _, number, sentence, _ in kept)


# --------------------------------------------------------------------------
# remote backends
# --------------------------------------------------------------------------


def _post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: float = 30
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class _RemoteBackend:
    """Shared plumbing: same prompt, and never raise -- degrade to extractive.

    A generation failure is a demo that still works with a visible caveat, not
    a 500 in the middle of a comparison. The caveat lands in `note`.
    """

    name = "remote"

    def __init__(self, fallback: ExtractiveBackend | None = None) -> None:
        self._fallback = fallback or ExtractiveBackend()

    def prompt_for(self, question: str, evidence: Sequence[Evidence]) -> str:
        return build_prompt(question, evidence)

    def generate(self, question: str, evidence: Sequence[Evidence]) -> Generation:
        try:
            text = self._call(self.prompt_for(question, evidence))
        except Exception as exc:  # noqa: BLE001 -- any failure means fall back
            degraded = self._fallback.generate(question, evidence)
            return Generation(
                text=degraded.text,
                backend=self._fallback.name,
                fell_back=True,
                note=(
                    f"The {self.name} backend was unavailable ({exc}), so this "
                    f"answer was produced extractively instead."
                ),
            )
        return Generation(text=text, backend=self.name)

    def _call(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaBackend(_RemoteBackend):
    name = "ollama"
    _response_key = "response"

    def __init__(
        self,
        host: str,
        model: str,
        temperature: float,
        timeout: float = 30,
        fallback: ExtractiveBackend | None = None,
    ) -> None:
        super().__init__(fallback)
        self._host = host.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout

    def _call(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        data = _post_json(f"{self._host}/api/generate", payload, timeout=self._timeout)
        return data[self._response_key]


class OpenAIBackend(_RemoteBackend):
    name = "openai"

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str,
        temperature: float,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30,
        fallback: ExtractiveBackend | None = None,
    ) -> None:
        super().__init__(fallback)
        # Read at construction so a missing key is reported once, in the note,
        # rather than per-question from deep inside a request.
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._temperature = temperature
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _call(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment")
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = _post_json(
            f"{self._base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self._timeout,
        )
        return data["choices"][0]["message"]["content"]


# The `llm:` block of config.yaml carries every knob all three backends need,
# so construction is a straight dispatch on one key.
def build_llm(cfg: dict[str, Any]) -> ExtractiveBackend | OllamaBackend | OpenAIBackend:
    backend = cfg["backend"]
    extractive = ExtractiveBackend(max_sentences=cfg.get("max_sentences", 6))
    if backend == "extractive":
        return extractive
    if backend == "ollama":
        return OllamaBackend(
            host=cfg["host"],
            model=cfg["model"],
            temperature=cfg.get("temperature", 0.1),
            fallback=extractive,
        )
    if backend == "openai":
        return OpenAIBackend(
            model=cfg["model"],
            temperature=cfg.get("temperature", 0.1),
            fallback=extractive,
        )
    raise ValueError(f"unknown llm backend {backend!r}")
