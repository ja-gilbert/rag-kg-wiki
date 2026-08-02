from __future__ import annotations

import dataclasses
import json
import time

import pytest

from approaches.base import ENVELOPE_FIELDS, Answer, Approach, PhaseTimer, estimate_tokens


def _answer(**overrides) -> Answer:
    defaults = dict(
        approach="rag",
        label="RAG",
        answer="Marcus Chen. [2]",
        evidence=[{"citation": "bug-903", "text": "Marcus Chen fixed BUG-903."}],
        citations=["bug-903"],
        trace=["Embedded the question.", "Took the top 5 chunks by cosine similarity."],
        detail={"scores": [0.61]},
        ms={"retrieve": 12.0, "total": 14.0},
        tokens_est=9,
        confident=True,
        note=None,
    )
    return Answer(**{**defaults, **overrides})


# --------------------------------------------------------------------------
# the envelope
# --------------------------------------------------------------------------


def test_the_envelope_has_exactly_the_documented_fields():
    # All three approaches and the frontend read this shape. Adding a field
    # without updating them together is the failure this test exists to catch.
    assert [f.name for f in dataclasses.fields(Answer)] == list(ENVELOPE_FIELDS)


def test_an_answer_exposes_every_field_as_a_dict():
    assert set(_answer().to_dict()) == set(ENVELOPE_FIELDS)


def test_an_answer_survives_a_json_round_trip():
    # It is served over /api/ask, so anything unserialisable is a bug here.
    assert json.loads(json.dumps(_answer().to_dict()))["approach"] == "rag"


def test_an_unconfident_answer_must_explain_itself():
    # Non-negotiable #7: confident=False is a feature, but only if the note
    # says why. A silent shrug is indistinguishable from a broken approach.
    with pytest.raises(ValueError, match="note"):
        _answer(confident=False, note=None)


def test_an_unconfident_answer_with_a_note_is_accepted():
    answer = _answer(confident=False, note="No path connects Beacon to revenue.")
    assert answer.confident is False
    assert answer.note


def test_a_confident_answer_needs_no_note():
    assert _answer(confident=True, note=None).note is None


# --------------------------------------------------------------------------
# phase timing
# --------------------------------------------------------------------------


def test_a_phase_is_recorded_in_milliseconds():
    timer = PhaseTimer()
    with timer.phase("retrieve"):
        time.sleep(0.02)
    assert timer.ms()["retrieve"] >= 15


def test_the_total_covers_every_phase():
    timer = PhaseTimer()
    with timer.phase("retrieve"):
        time.sleep(0.01)
    with timer.phase("generate"):
        time.sleep(0.01)
    ms = timer.ms()
    assert ms["total"] >= ms["retrieve"] + ms["generate"]


def test_a_phase_entered_twice_accumulates():
    timer = PhaseTimer()
    for _ in range(2):
        with timer.phase("retrieve"):
            time.sleep(0.01)
    assert timer.ms()["retrieve"] >= 15


def test_a_total_is_reported_even_when_nothing_was_phased():
    assert PhaseTimer().ms() == {"total": pytest.approx(0, abs=50)}


def test_a_phase_that_raises_is_still_timed():
    # Otherwise a failing approach reports ms={} and the scoreboard lies.
    timer = PhaseTimer()
    with pytest.raises(RuntimeError), timer.phase("retrieve"):
        raise RuntimeError("boom")
    assert "retrieve" in timer.ms()


# --------------------------------------------------------------------------
# evidence size
# --------------------------------------------------------------------------


def test_no_evidence_estimates_no_tokens():
    assert estimate_tokens([]) == 0


def test_the_estimate_grows_with_the_evidence():
    assert estimate_tokens(["word " * 100]) > estimate_tokens(["word " * 10])


def test_the_estimate_is_roughly_four_characters_per_token():
    assert estimate_tokens(["x" * 400]) == 100


# --------------------------------------------------------------------------
# the approach contract
# --------------------------------------------------------------------------


def test_the_base_approach_refuses_to_answer():
    with pytest.raises(NotImplementedError):
        Approach().answer("who fixed BUG-903?")
