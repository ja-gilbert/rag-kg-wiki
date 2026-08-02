from __future__ import annotations

import re

# Shared by the extractive generator and the wiki's catalogue scan. Both are
# doing the same job -- deciding which text a question is about -- and having
# two stopword lists drift apart would make their behaviour differ for reasons
# nobody chose.

STOPWORDS = frozenset(
    """
    a an and are as at be by did do does for from has have how in into is it its
    of on or that the their there they this to was were what when where which
    who whom whose why will with
    """.split()
)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")


def terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)} - STOPWORDS


# Crude suffix stripping, not a real stemmer. It exists because the wiki's
# catalogue scan matches a question against page titles, and exact matching
# misses "rotating credentials" against a page called "Credential Rotation
# Standard" -- which is the page that answers it. Longest suffix first, and a
# four-character floor so "keys" -> "key" but "does" survives intact.
_SUFFIXES = ("ations", "ation", "ions", "ing", "ies", "ion", "es", "ed", "s")


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def stems(text: str) -> set[str]:
    return {stem(term) for term in terms(text)}
