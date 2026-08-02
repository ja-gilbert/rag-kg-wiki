from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from core.llm import (
    NO_EVIDENCE_TEXT,
    Evidence,
    ExtractiveBackend,
    OllamaBackend,
    OpenAIBackend,
    build_llm,
    build_prompt,
)

# A port nothing listens on. Connecting here fails immediately with "connection
# refused" rather than hanging, so the fallback tests stay fast and offline.
_CLOSED_PORT = "http://127.0.0.1:1"

_EVIDENCE = [
    Evidence(citation="svc-atlas", text="Atlas depends on Ember for search."),
    Evidence(citation="bug-903", text="BUG-903 affects Ember. Marcus Chen fixed BUG-903."),
]


# --------------------------------------------------------------------------
# the shared prompt contract
# --------------------------------------------------------------------------


def test_the_prompt_numbers_every_evidence_block():
    prompt = build_prompt("who fixed it?", _EVIDENCE)
    assert "[1]" in prompt
    assert "[2]" in prompt


def test_the_prompt_carries_each_blocks_citation():
    prompt = build_prompt("who fixed it?", _EVIDENCE)
    assert "svc-atlas" in prompt
    assert "bug-903" in prompt


def test_the_prompt_contains_the_question():
    assert "who fixed it?" in build_prompt("who fixed it?", _EVIDENCE)


def test_the_prompt_instructs_the_model_to_cite():
    assert "cite" in build_prompt("who fixed it?", _EVIDENCE).lower()


def test_the_prompt_instructs_the_model_to_admit_missing_evidence():
    # Non-negotiable #7: the architecture must be able to say "I don't know".
    prompt = build_prompt("who fixed it?", _EVIDENCE).lower()
    assert "does not contain" in prompt


def test_every_backend_builds_the_identical_prompt():
    # Non-negotiable #5: any on-screen difference must come from the evidence
    # the architecture found, never from a backend getting a better prompt.
    extractive = ExtractiveBackend()
    ollama = OllamaBackend(host=_CLOSED_PORT, model="llama3.2", temperature=0.1)
    openai = OpenAIBackend(model="gpt-4o-mini", temperature=0.1, api_key="k")
    prompts = {b.prompt_for("who fixed it?", _EVIDENCE) for b in (extractive, ollama, openai)}
    assert len(prompts) == 1


# --------------------------------------------------------------------------
# the extractive backend
# --------------------------------------------------------------------------


def test_extractive_answers_with_the_sentence_that_matches_the_question():
    generation = ExtractiveBackend().generate("who fixed BUG-903?", _EVIDENCE)
    assert "Marcus Chen fixed BUG-903." in generation.text


def test_extractive_cites_the_block_each_sentence_came_from():
    generation = ExtractiveBackend().generate("who fixed BUG-903?", _EVIDENCE)
    assert re.search(r"Marcus Chen fixed BUG-903\.\s*\[2\]", generation.text)


def test_extractive_never_emits_a_word_absent_from_the_evidence():
    # The whole point of this backend: it cannot hallucinate and cannot
    # paraphrase, because it only ever moves existing sentences around.
    generation = ExtractiveBackend().generate("who fixed BUG-903?", _EVIDENCE)
    prose = re.sub(r"\[\d+\]", "", generation.text)  # citations are ours, not the corpus's
    said = set(re.findall(r"[A-Za-z0-9-]+", prose))
    available = set(re.findall(r"[A-Za-z0-9-]+", " ".join(e.text for e in _EVIDENCE)))
    assert said <= available


def test_extractive_keeps_at_most_max_sentences():
    evidence = [
        Evidence(citation="d", text=" ".join(f"Ember sentence number {n}." for n in range(10)))
    ]
    generation = ExtractiveBackend(max_sentences=3).generate("Ember", evidence)
    assert generation.text.count("Ember sentence number") == 3


def test_extractive_reports_itself_as_the_producing_backend():
    generation = ExtractiveBackend().generate("who fixed BUG-903?", _EVIDENCE)
    assert generation.backend == "extractive"
    assert generation.fell_back is False


def test_extractive_says_so_plainly_when_there_is_no_evidence():
    generation = ExtractiveBackend().generate("who fixed BUG-903?", [])
    assert generation.text == NO_EVIDENCE_TEXT


def test_extractive_sentences_appear_in_evidence_order():
    # Ranking picks which sentences survive; reading order stays the order the
    # retriever handed them over, so the citations count upwards.
    generation = ExtractiveBackend(max_sentences=2).generate("Ember BUG-903", _EVIDENCE)
    assert generation.text.index("[1]") < generation.text.index("[2]")


# --------------------------------------------------------------------------
# remote backends: fallback rather than failure
# --------------------------------------------------------------------------


def test_ollama_falls_back_to_extractive_when_the_host_is_unreachable():
    backend = OllamaBackend(host=_CLOSED_PORT, model="llama3.2", temperature=0.1, timeout=1)
    generation = backend.generate("who fixed BUG-903?", _EVIDENCE)
    assert generation.fell_back is True
    assert generation.backend == "extractive"
    assert "Marcus Chen fixed BUG-903." in generation.text


def test_a_fallback_explains_itself_in_its_note():
    backend = OllamaBackend(host=_CLOSED_PORT, model="llama3.2", temperature=0.1, timeout=1)
    note = backend.generate("who fixed BUG-903?", _EVIDENCE).note
    assert note is not None
    assert "ollama" in note.lower()


def test_openai_falls_back_when_no_api_key_is_configured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    generation = OpenAIBackend(model="gpt-4o-mini", temperature=0.1).generate("who?", _EVIDENCE)
    assert generation.fell_back is True
    assert generation.backend == "extractive"
    assert "OPENAI_API_KEY" in (generation.note or "")


def test_openai_reads_its_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert OpenAIBackend(model="gpt-4o-mini", temperature=0.1).api_key == "sk-test"


# --------------------------------------------------------------------------
# remote backends: the success path, against a local stub server
# --------------------------------------------------------------------------


class _StubHandler(BaseHTTPRequestHandler):
    """Answers both the ollama and the openai response shapes."""

    def do_POST(self):  # noqa: N802 -- BaseHTTPRequestHandler's spelling
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
        if "chat/completions" in self.path:
            payload = {"choices": [{"message": {"content": "Marcus Chen. [2]"}}]}
        else:
            payload = {"response": "Marcus Chen. [2]"}
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):
        pass  # keep pytest output pristine


@pytest.fixture
def stub_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _url(server) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}"


def test_ollama_returns_the_models_text_when_the_host_answers(stub_server):
    backend = OllamaBackend(host=_url(stub_server), model="llama3.2", temperature=0.1, timeout=5)
    generation = backend.generate("who fixed BUG-903?", _EVIDENCE)
    assert generation.text == "Marcus Chen. [2]"
    assert generation.backend == "ollama"
    assert generation.fell_back is False


def test_ollama_sends_the_configured_model_and_temperature(stub_server):
    backend = OllamaBackend(host=_url(stub_server), model="llama3.2", temperature=0.1, timeout=5)
    backend.generate("who fixed BUG-903?", _EVIDENCE)
    body = stub_server.requests[0]["body"]
    assert body["model"] == "llama3.2"
    assert body["options"]["temperature"] == 0.1
    assert body["stream"] is False


def test_ollama_sends_the_shared_prompt(stub_server):
    backend = OllamaBackend(host=_url(stub_server), model="llama3.2", temperature=0.1, timeout=5)
    backend.generate("who fixed BUG-903?", _EVIDENCE)
    assert stub_server.requests[0]["body"]["prompt"] == build_prompt("who fixed BUG-903?", _EVIDENCE)


def test_openai_returns_the_models_text_when_the_api_answers(stub_server):
    backend = OpenAIBackend(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key="sk-test",
        base_url=_url(stub_server),
        timeout=5,
    )
    generation = backend.generate("who fixed BUG-903?", _EVIDENCE)
    assert generation.text == "Marcus Chen. [2]"
    assert generation.backend == "openai"


def test_openai_authorises_with_the_bearer_token(stub_server):
    backend = OpenAIBackend(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key="sk-test",
        base_url=_url(stub_server),
        timeout=5,
    )
    backend.generate("who fixed BUG-903?", _EVIDENCE)
    headers = stub_server.requests[0]["headers"]
    assert headers["Authorization"] == "Bearer sk-test"


def test_a_remote_backend_falls_back_when_the_response_shape_is_wrong(stub_server):
    # A 200 carrying JSON we cannot read is a failure like any other, and must
    # degrade to extractive rather than raising into the request handler.
    backend = OllamaBackend(host=_url(stub_server), model="m", temperature=0.1, timeout=5)
    backend._response_key = "nope"
    generation = backend.generate("who fixed BUG-903?", _EVIDENCE)
    assert generation.fell_back is True
    assert generation.backend == "extractive"


# --------------------------------------------------------------------------
# construction from config
# --------------------------------------------------------------------------


def test_build_llm_selects_the_configured_backend():
    cfg = {"backend": "extractive", "model": "llama3.2", "host": _CLOSED_PORT,
           "temperature": 0.1, "max_sentences": 6}
    assert build_llm(cfg).name == "extractive"
    assert build_llm({**cfg, "backend": "ollama"}).name == "ollama"
    assert build_llm({**cfg, "backend": "openai"}).name == "openai"


def test_build_llm_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown llm backend"):
        build_llm({"backend": "gpt-5-ultra"})


def test_build_llm_honours_max_sentences():
    backend = build_llm({"backend": "extractive", "max_sentences": 2})
    evidence = [Evidence(citation="d", text=" ".join(f"Ember number {n}." for n in range(6)))]
    assert backend.generate("Ember", evidence).text.count("Ember number") == 2
