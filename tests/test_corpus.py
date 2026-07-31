from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from core.corpus import load_corpus, split_sentences

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


_FORBIDDEN_LINE_STARTS = ("---", "#")
_FORBIDDEN_SUBSTRINGS = ("relations:",)


def _structural_violations(path: Path) -> list[str]:
    """Non-negotiable #1: every edge must be extracted from prose, so the raw
    corpus may not carry frontmatter, markdown or declared relations."""
    violations = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith(_FORBIDDEN_LINE_STARTS):
            violations.append(f"{path.name}:{n}: machine-readable structure: {line!r}")
        if any(s in line for s in _FORBIDDEN_SUBSTRINGS):
            violations.append(f"{path.name}:{n}: declared relations: {line!r}")
    return violations


def test_raw_corpus_contains_only_plain_text_files():
    files = sorted(p for p in RAW.iterdir() if p.is_file())
    assert files, "data/raw is empty"
    assert not [f.name for f in files if f.suffix != ".txt"]


def test_raw_corpus_carries_no_machine_readable_structure():
    violations = [v for f in sorted(RAW.glob("*.txt")) for v in _structural_violations(f)]
    assert not violations, "\n".join(violations)


def test_the_structure_guard_catches_a_planted_violation(tmp_path):
    # The guard above passes on arrival, so it proves nothing until we show it
    # can fail. Plant every violation it is meant to catch.
    planted = tmp_path / "svc-fake.txt"
    planted.write_text(
        "---\ntitle: Fake\nrelations:\n  - owned_by: Platform Team\n---\n\n"
        "# Heading\n\nProse.\n",
        encoding="utf-8",
    )
    violations = _structural_violations(planted)
    assert any("machine-readable structure" in v for v in violations)
    assert any("declared relations" in v for v in violations)


def test_load_corpus_returns_every_document():
    assert len(load_corpus(RAW)) == 39


def test_every_document_has_its_core_fields_populated():
    for doc in load_corpus(RAW):
        assert doc.doc_id, f"{doc.path}: empty doc_id"
        assert doc.title, f"{doc.path}: empty title"
        assert doc.doc_type, f"{doc.path}: empty doc_type"
        assert doc.body, f"{doc.path}: empty body"


def test_body_excludes_the_three_header_lines():
    # The title is compared as a whole line, not as a prefix: person- documents
    # open by naming the person, so `body.startswith(title)` is true for them
    # even when the header was stripped correctly.
    for doc in load_corpus(RAW):
        first_line = doc.path.read_text(encoding="utf-8").splitlines()[0]
        assert "Last updated:" not in doc.body, f"{doc.doc_id}: date line leaked"
        assert "Meridian Systems  |" not in doc.body, f"{doc.doc_id}: banner leaked"
        assert doc.body.splitlines()[0] != first_line, f"{doc.doc_id}: title leaked"


def test_doc_type_comes_from_the_filename_prefix():
    by_id = {d.doc_id: d for d in load_corpus(RAW)}
    assert by_id["svc-atlas"].doc_type == "service"
    assert by_id["inc-2088"].doc_type == "incident"
    assert by_id["person-marcus-chen"].doc_type == "person"
    # arch- and ref- both collapse to reference material.
    assert by_id["arch-overview"].doc_type == "reference"
    assert by_id["ref-glossary"].doc_type == "reference"


def test_unknown_filename_prefix_raises_naming_the_offender(tmp_path):
    stray = tmp_path / "memo-q3-planning.txt"
    stray.write_text(
        "Q3 Planning\nMeridian Systems  |  Memo\nLast updated: 2026-01-01\n\nBody.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        load_corpus(tmp_path)
    message = str(excinfo.value)
    assert "memo" in message, "error should name the unrecognised prefix"
    assert "memo-q3-planning" in message, "error should name the offending file"


def test_title_and_date_come_from_the_header():
    by_id = {d.doc_id: d for d in load_corpus(RAW)}
    atlas = by_id["svc-atlas"]
    assert atlas.title == "Atlas API Gateway"
    assert atlas.date == dt.date(2025, 11, 4)


def test_split_sentences_yields_whole_single_line_sentences():
    atlas = {d.doc_id: d for d in load_corpus(RAW)}["svc-atlas"]
    sentences = split_sentences(atlas.body)
    assert sentences
    for s in sentences:
        assert "\n" not in s, f"hard-wrap newline survived: {s!r}"
        assert s == s.strip(), f"untrimmed sentence: {s!r}"
        assert s[0].isupper() or s[0].isdigit(), f"starts mid-clause: {s!r}"


def test_no_sentence_anywhere_in_the_corpus_contains_a_newline():
    # The corpus is hard-wrapped at ~95 columns, so this is the assertion that
    # actually proves intra-paragraph newlines were collapsed.
    for doc in load_corpus(RAW):
        for s in split_sentences(doc.body):
            assert "\n" not in s, f"{doc.doc_id}: {s!r}"


def test_split_sentences_does_not_break_on_an_abbreviation():
    # Real prose from person-elena-vasquez.txt -- the corpus's only
    # abbreviation, and hard-wrapped mid-sentence into the bargain.
    body = (
        "Dr. Elena Vasquez is the chief technology officer of Meridian Systems"
        " and has been with the\ncompany since its second year."
    )
    assert split_sentences(body) == [
        "Dr. Elena Vasquez is the chief technology officer of Meridian Systems"
        " and has been with the company since its second year."
    ]
