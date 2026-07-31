from __future__ import annotations

from pathlib import Path

import pytest

from core.chunking import chunk_documents
from core.config import load_config
from core.corpus import load_corpus

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"


def test_load_config_reads_every_section():
    cfg = load_config(CONFIG)
    assert set(cfg) >= {
        "paths",
        "embedding",
        "chunking",
        "rag",
        "kg",
        "wiki",
        "llm",
        "server",
    }


def test_chunking_section_drives_chunk_documents_directly():
    # Step 2 promised strategies "selectable from config.yaml" but nothing read
    # the file. This is the assertion that makes that true rather than notional.
    cfg = load_config(CONFIG)
    corpus = load_corpus(ROOT / "data" / "raw")
    chunks = chunk_documents(corpus, cfg["chunking"])
    assert chunks
    assert {c.doc_id for c in chunks} == {d.doc_id for d in corpus}


def test_scalar_types_survive_the_yaml_round_trip():
    # Everything downstream indexes these without coercing, so a quoted int in
    # config.yaml would surface much later as a confusing slice or range error.
    cfg = load_config(CONFIG)
    assert isinstance(cfg["chunking"]["window"], int)
    assert isinstance(cfg["rag"]["top_k"], int)
    assert isinstance(cfg["rag"]["alpha"], float)
    assert isinstance(cfg["server"]["reload"], bool)


def test_missing_config_file_raises_naming_the_path(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_config(tmp_path / "nope.yaml")
    assert "nope.yaml" in str(excinfo.value)
