from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.corpus import Document, split_sentences

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

# Defaults mirror the `chunking:` block in config.yaml, so a caller may pass a
# partial dict and still get the documented behaviour.
_DEFAULTS = {"window": 3, "stride": 2, "size": 900, "overlap": 150}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str


def _paragraph_texts(doc: Document) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_BREAK.split(doc.body) if p.strip()]


def _sentence_window_texts(doc: Document, window: int, stride: int) -> list[str]:
    sentences = split_sentences(doc.body)
    texts: list[str] = []
    for start in range(0, len(sentences), stride):
        group = sentences[start : start + window]
        if not group:
            break
        texts.append(" ".join(group))
        # Once a window reaches the final sentence, further strides could only
        # emit tails already contained in this one.
        if start + window >= len(sentences):
            break
    return texts


def _fixed_texts(doc: Document, size: int, overlap: int) -> list[str]:
    # Deliberately blunt: this walks raw characters and cuts wherever the window
    # lands, mid-word and mid-clause. Keeping it dumb is the point -- it is the
    # control case that shows how much chunking strategy alone moves retrieval.
    body = doc.body
    step = size - overlap
    if step <= 0:
        raise ValueError(
            f"chunking overlap ({overlap}) must be smaller than size ({size}), "
            "or the window never advances"
        )
    texts: list[str] = []
    for start in range(0, len(body), step):
        window = body[start : start + size].strip()
        if window:
            texts.append(window)
        if start + size >= len(body):
            break
    return texts


def _texts_for(doc: Document, cfg: dict[str, Any]) -> list[str]:
    strategy = cfg["strategy"]
    if strategy == "paragraph":
        return _paragraph_texts(doc)
    if strategy == "sentence_window":
        return _sentence_window_texts(
            doc,
            cfg.get("window", _DEFAULTS["window"]),
            cfg.get("stride", _DEFAULTS["stride"]),
        )
    if strategy == "fixed":
        return _fixed_texts(
            doc,
            cfg.get("size", _DEFAULTS["size"]),
            cfg.get("overlap", _DEFAULTS["overlap"]),
        )
    raise ValueError(f"unknown chunking strategy {strategy!r}")


def chunk_documents(documents: list[Document], cfg: dict[str, Any]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        for n, text in enumerate(_texts_for(doc, cfg)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{n}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    text=text,
                )
            )
    return chunks
