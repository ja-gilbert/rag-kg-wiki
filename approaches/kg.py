from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from approaches.base import Answer, Approach, PhaseTimer, estimate_tokens
from approaches.planner import QuestionPlan, plan_question
from core.llm import Evidence
from kgraph.elements import Neighbour, Path
from kgraph.graph import KnowledgeGraph
from kgraph.ontology import Gazetteer, Ontology

# Scoring weights from docs/ARCHITECTURE.md. A path is a better answer when it
# used the relations the question asked about, when it routed through the kinds
# of thing the question named, and when it did so in fewer hops.
_CUE_EDGE = 2.5
_NAMED_INTERMEDIATE = 1.5
_PER_HOP = -0.6


@dataclass(frozen=True)
class Candidate:
    entity: str
    score: float
    path: Path


class KgApproach(Approach):
    """Plan the question, walk the graph, answer with the path that got there.

    The path is the product, not a by-product: it is what RAG cannot produce,
    because no passage contains it. So it travels all the way through -- into
    the prompt as one block per route, into `detail` for the graph tab to
    highlight, and into the citations as every document it crossed.
    """

    name = "kg"
    label = "Knowledge graph"

    def __init__(
        self, graph: KnowledgeGraph, ontology: Ontology, llm: Any, cfg: dict[str, Any]
    ) -> None:
        self._graph = graph
        self._ontology = ontology
        self._llm = llm
        self._gazetteer = Gazetteer(ontology)
        self._max_hops = cfg.get("max_hops", 4)
        self._max_answers = cfg.get("max_answers", 6)

    def answer(self, question: str) -> Answer:
        timer = PhaseTimer()
        with timer.phase("plan"):
            plan = plan_question(question, self._ontology, self._gazetteer)

        with timer.phase("search"):
            neighbours = candidates = ()
            if plan.hops is not None:
                neighbours = self._neighbourhood(plan)
                blocks = [_neighbour_dict(n) for n in neighbours]
            else:
                candidates = self._candidates(plan)
                blocks = [_candidate_dict(c) for c in candidates]

        with timer.phase("generate"):
            # No path means no evidence, and an empty prompt is precisely where
            # a live model answers from its own memory instead of from the
            # corpus. The graph declining is the demo; don't hand it away.
            generation = self._llm.generate(question, _evidence(blocks)) if blocks else None

        detail = {
            "query": "neighbourhood" if plan.hops is not None else "path",
            "seeds": list(plan.seeds),
            "answer_type": plan.answer_type,
            "named_types": list(plan.named_types),
            "cues": list(plan.cues),
            "max_hops": self._max_hops,
        }
        if plan.hops is not None:
            detail |= {"hops": plan.hops, "neighbours": blocks}
        else:
            best = candidates[0] if candidates else None
            detail |= {
                "candidates": blocks,
                "answer_entity": best.entity if best else None,
                "score": round(best.score, 2) if best else None,
                "path": blocks[0] if blocks else None,
            }

        confident, note = self._verdict(plan, candidates, blocks, generation)
        return Answer(
            approach=self.name,
            label=self.label,
            answer=generation.text if generation else self._no_path_text(plan),
            evidence=blocks,
            citations=_citations(blocks),
            trace=self._trace(plan, blocks),
            detail=detail,
            ms=timer.ms(),
            tokens_est=estimate_tokens(b["text"] for b in blocks),
            confident=confident,
            note=note,
        )

    # --- search --------------------------------------------------------------

    def _neighbourhood(self, plan: QuestionPlan) -> tuple[Neighbour, ...]:
        types = {plan.answer_type} if plan.answer_type else None
        found: dict[str, Neighbour] = {}
        for seed in plan.seeds:
            for neighbour in self._graph.neighbors_within(seed, plan.hops or 0, types):
                if neighbour.name in plan.seeds:
                    continue
                # Nearest wins when two seeds reach the same entity.
                if neighbour.name not in found or neighbour.distance < found[neighbour.name].distance:
                    found[neighbour.name] = neighbour
        return tuple(sorted(found.values(), key=lambda n: (n.distance, n.name)))

    def _candidates(self, plan: QuestionPlan) -> tuple[Candidate, ...]:
        types = {plan.answer_type} if plan.answer_type else None
        best: dict[str, Candidate] = {}
        for seed in plan.seeds:
            for neighbour in self._graph.neighbors_within(seed, self._max_hops, types):
                if neighbour.name in plan.seeds:
                    continue
                # Every route, not just the shortest: the whole point of
                # scoring is that a longer path through the relations the
                # question named beats a short one through relations it didn't.
                for path in self._graph.paths_between(seed, neighbour.name, self._max_hops):
                    candidate = Candidate(neighbour.name, self._score(path, plan), path)
                    if neighbour.name not in best or candidate.score > best[neighbour.name].score:
                        best[neighbour.name] = candidate
        ranked = sorted(best.values(), key=lambda c: (-c.score, c.entity))
        return tuple(ranked[: self._max_answers])

    def _score(self, path: Path, plan: QuestionPlan) -> float:
        """How much of the question this route explains, per hop spent.

        Counted once per *distinct* cue and per distinct intermediate type,
        not once per edge as docs/ARCHITECTURE.md suggests. Per-edge scoring
        pays repeatedly for one word in the question, and on this corpus that
        loses the Beacon question outright: a four-hop chain of three
        `depends on` edges through three Services scores 9.6, beating the
        route the question actually describes -- depends_on then caused_by,
        through a Service and an Incident -- at 6.2. Matching two different
        cues has to beat repeating one.
        """
        cues = set(plan.cues)
        # The answer type is excluded: it names the destination, not a waypoint.
        # Counting it rewards routing through another entity of the same type,
        # and on this corpus that reaches Data Retention Policy v2 -- the
        # planted superseded policy -- by way of the v3 that replaced it.
        named = set(plan.named_types) - {plan.answer_type}
        matched = {hop.edge.predicate for hop in path.hops} & cues
        crossed = {self._graph.type_of(node) for node in path.nodes[1:-1]} & named
        return _PER_HOP * len(path.hops) + _CUE_EDGE * len(matched) + _NAMED_INTERMEDIATE * len(crossed)

    # --- reporting -----------------------------------------------------------

    def _verdict(
        self,
        plan: QuestionPlan,
        candidates: tuple[Candidate, ...],
        blocks: list[dict[str, Any]],
        generation: Any,
    ) -> tuple[bool, str | None]:
        if not blocks:
            return False, self._no_path_text(plan)
        # Two routes scoring identically is a real ambiguity, and saying so is
        # worth more than picking whichever sorted first.
        if len(candidates) > 1 and candidates[0].score == candidates[1].score:
            return False, (
                f"The graph found two equally good routes, to "
                f"{candidates[0].entity} and {candidates[1].entity}. "
                "Neither is better supported than the other."
            )
        return True, (generation.note if generation and generation.note else None)

    def _no_path_text(self, plan: QuestionPlan) -> str:
        if not plan.seeds:
            return (
                "No entity in this question matches anything in the ontology, so "
                "there is nowhere in the graph to start looking."
            )
        seeds = ", ".join(plan.seeds)
        target = f"any {plan.answer_type}" if plan.answer_type else "anything"
        return (
            f"No path of {self._max_hops} hops or fewer connects {seeds} to "
            f"{target} in the graph."
        )

    def _trace(self, plan: QuestionPlan, blocks: list[dict[str, Any]]) -> list[str]:
        steps = [
            f"Matched the question against the gazetteer: "
            f"{', '.join(plan.seeds) if plan.seeds else 'no known entity'}.",
            f"Read the answer type as {plan.answer_type or 'unconstrained'}.",
            f"Took {', '.join(plan.cues) if plan.cues else 'no'} relation(s) as cues "
            "from the question's verbs.",
        ]
        if plan.hops is not None:
            steps.append(f"Ran a neighbourhood query to {plan.hops} hop(s).")
            steps.append(f"Found {len(blocks)} entity(ies) of that type in range.")
        else:
            steps.append(
                f"Walked the graph to {self._max_hops} hops, scoring each route by "
                "how well it matched those cues."
            )
            steps.append(
                f"Kept {len(blocks)} candidate(s)."
                if blocks
                else "Found no route at all, so there is nothing to report."
            )
        return steps


# --- serialisation -----------------------------------------------------------


def _hop_dicts(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "source": hop.source,
            "target": hop.target,
            "label": hop.label,
            "forward": hop.forward,
            "predicate": hop.edge.predicate,
            # Non-negotiable #8: the sentence travels with the edge, always.
            "sentence": hop.sentence,
            "doc_id": hop.doc_id,
        }
        for hop in path.hops
    ]


def _block(entity: str, path: Path, **extra: Any) -> dict[str, Any]:
    hops = _hop_dicts(path)
    return {
        "entity": entity,
        "render": path.render(),
        "nodes": list(path.nodes),
        "hops": hops,
        "doc_ids": list(dict.fromkeys(h["doc_id"] for h in hops)),
        # What the generator sees: the chain, then the sentence justifying each
        # link of it. Kept whole rather than split per hop, because the chain is
        # the thing the graph found and splitting it would hand the model the
        # same shape of evidence RAG hands it.
        "text": "\n".join([path.render(), *(f'"{h["sentence"]}"' for h in hops)]),
        **extra,
    }


def _candidate_dict(candidate: Candidate) -> dict[str, Any]:
    return _block(candidate.entity, candidate.path, score=round(candidate.score, 2))


def _neighbour_dict(neighbour: Neighbour) -> dict[str, Any]:
    return _block(
        neighbour.name,
        neighbour.via,
        type=neighbour.type,
        distance=neighbour.distance,
        via=neighbour.via.render(),
    )


def _evidence(blocks: list[dict[str, Any]]) -> list[Evidence]:
    return [Evidence(citation=", ".join(b["doc_ids"]), text=b["text"]) for b in blocks]


def _citations(blocks: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(doc_id for b in blocks for doc_id in b["doc_ids"]))
