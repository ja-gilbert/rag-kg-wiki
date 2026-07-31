from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

# Document type comes from the filename prefix, not from anything inside the
# file. The corpus is deliberately structure-free prose (non-negotiable #1), so
# a naming convention is the only classification signal a real corpus would give
# us. `arch-` and `ref-` both collapse to reference material.
DOC_TYPE_BY_PREFIX = {
    "svc": "service",
    "inc": "incident",
    "bug": "bug",
    "pol": "policy",
    "run": "runbook",
    "team": "team",
    "person": "person",
    "vendor": "vendor",
    "arch": "reference",
    "ref": "reference",
}

_DATE_MARKER = "Last updated:"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    doc_type: str
    date: dt.date | None
    body: str
    path: Path


def doc_type_for(doc_id: str) -> str:
    prefix = doc_id.split("-", 1)[0]
    try:
        return DOC_TYPE_BY_PREFIX[prefix]
    except KeyError:
        # Deliberately fatal. Defaulting to "document" would let a misnamed or
        # newly-added file join the corpus untyped, and every downstream
        # approach keys off doc_type.
        known = ", ".join(sorted(DOC_TYPE_BY_PREFIX))
        raise ValueError(
            f"{doc_id}: unrecognised filename prefix {prefix!r}. "
            f"Known prefixes: {known}. Add it to DOC_TYPE_BY_PREFIX."
        ) from None


def _parse(path: Path) -> Document:
    # read_text applies universal newlines, so a CRLF checkout still splits
    # cleanly here even though .gitattributes pins the corpus to LF.
    header, _, body = path.read_text(encoding="utf-8").partition("\n\n")
    lines = header.splitlines()

    date = None
    for line in lines:
        if line.startswith(_DATE_MARKER):
            date = dt.date.fromisoformat(line[len(_DATE_MARKER) :].strip())

    return Document(
        doc_id=path.stem,
        title=lines[0].strip(),
        doc_type=doc_type_for(path.stem),
        date=date,
        body=body.strip(),
        path=path,
    )


def load_corpus(raw_dir: Path) -> list[Document]:
    return [_parse(p) for p in sorted(Path(raw_dir).glob("*.txt"))]


# Abbreviations whose period does not end a sentence. Today's corpus contains
# exactly one ("Dr."), but this splitter is the foundation chunking, relation
# extraction and extractive answering all read through, so the set is slightly
# wider than the current data strictly needs.
_ABBREVIATIONS = frozenset(
    {
        "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "St",
        "vs", "etc", "e.g", "i.e", "approx", "No", "Inc", "Ltd", "Co",
    }
)

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"""([.!?]["')\]]*)\s+""")
_WORD_BEFORE_END = re.compile(r"""([A-Za-z][A-Za-z.]*)[.!?]["')\]]*$""")


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        # The corpus is hard-wrapped at ~95 columns, so a newline inside a
        # paragraph is a typesetting artefact rather than a boundary. Collapse
        # it first or every sentence arrives in ~95-character pieces.
        flat = " ".join(paragraph.split())
        if not flat:
            continue

        start = 0
        for match in _SENTENCE_END.finditer(flat):
            candidate = flat[start : match.end(1)]
            trailing = _WORD_BEFORE_END.search(candidate)
            if trailing and trailing.group(1).rstrip(".") in _ABBREVIATIONS:
                continue  # "Dr." -- swallow the period, keep accumulating
            sentences.append(candidate.strip())
            start = match.end()

        tail = flat[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences
