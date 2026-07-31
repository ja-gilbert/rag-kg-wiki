# Architecture

The design spec. `docs/IMPLEMENTATION.md` is the build order; this is the
reasoning behind each piece and the contracts they have to honour.

## The shared spine

All three approaches consume the same corpus and return the same envelope:

```
data/raw/*.txt          <- plain text in
    │
    ├──► core.corpus.load_corpus()  ──►  list[Document]
    │
    ├──► RAG    chunk ──► embed ──► VectorStore
    ├──► GRAPH  extract triples ──► KnowledgeGraph
    └──► WIKI   compile ──► wiki/pages/*.md + index.md + log.md
                                    │
                              all three ──► Answer ──► /api/ask
```

### The `Answer` contract

```python
approach     "rag" | "kg" | "wiki"
label        display name
answer       the generated text
evidence     list[dict] -- chunks, paths, or pages
citations    source ids
trace        what it actually did, step by step, in plain English
detail       approach-specific payload (plans, paths, scores, highlights)
ms           per-phase timing plus total
tokens_est   evidence size handed to the generator
confident    bool
note         the honest caveat when confident is False
```

Two rules follow from this, and they're the reason the comparison holds:

**Same generator, same prompt.** Every approach gets the same LLM backend and
the same prompt contract — numbered evidence blocks, a question, an instruction
to cite. Any difference on screen is a difference in *what evidence the
architecture found*, not in who got the better model.

**Same envelope, same layout.** The three UI columns render identical sections
in identical order. Bespoke layout for one approach would make it look better
for reasons unrelated to its merits.

---

## RAG

```
question ──► embed ──► cosine vs. N chunk vectors ──► top-k ──► generate
```

Exact cosine over a dense numpy matrix. One matrix multiply, no FAISS, no ANN
index — at 39 documents an approximate index would be a lie about where the
complexity lives, and the whole search should be readable in one screen. BM25
sits alongside it so the UI can show lexical and semantic retrieval on the same
query, plus a normalised linear fusion of the two.

Three things this design makes visible that most RAG demos hide:

**Chunking is a decision, not a detail.** Three strategies, switchable in
config. `fixed` splits sentences mid-clause and you get to watch retrieval
quality drop.

**The embedding space is swappable.** Neural, LSA, or deliberately meaningless.
The `hash` backend exists so you can see what "retrieval" looks like when the
vectors carry no semantics — the machinery runs perfectly and returns nonsense.

**The similarity score is always shown.** A `low_confidence_threshold` flags
weak retrievals. Without it, a nearest-neighbour shrug is indistinguishable from
a real hit — which is RAG's most consequential failure mode, because vector
search *always returns something*. A similarity score is never zero.

Strengths and weaknesses both follow directly from the design. Nothing is
precomputed beyond the index, so new documents are one re-embed away from being
answerable. But it retrieves *passages*, so any answer requiring a join across
documents is only findable if some passage happens to contain the join — and it
has no notion of "two hops away", "all of", or "none of".

---

## Knowledge graph

### Extraction

Classic rules-and-gazetteer, deliberately. An LLM extractor would be less code
and more magic; the point here is being able to click an edge and see the
sentence it came from.

```
1. Gazetteer     canonical entities + aliases from data/ontology.yaml,
                 longest-match-wins over each sentence
2. Patterns      "{s} is owned by {o}" compiled into a regex whose slots are
                 alternations over every known entity surface form
3. Anaphora      bare-subject patterns ("The team owns Beacon") resolve their
                 subject to the document's primary entity
4. Provenance    every triple carries doc_id, sentence, and matching pattern
```

Tuning the ontology until zero entities are orphaned *is* the work of this
approach, and it should feel like it. That asymmetry — RAG builds in seconds,
the graph needs an ontology someone thought about — is a genuine finding, not an
artifact of the demo. Report it in the build summary.

Add domain/range type constraints to relations and reject triples whose
endpoints don't type-check. `owned_by` takes a Service and a Team; anything else
is an extraction bug, and catching it structurally beats catching it by eye.

### Query planning

```
1. Seeds          gazetteer match against the question
2. Answer type    "who" -> Person, "which policy" -> Policy, ...
3. Predicate cues "fixed" -> fixed_by, "depends" -> depends_on, ...
4. Search         BFS from seeds; score each candidate of the target type by
                  how well its justifying path matches the cues
```

Suggested scoring: `+2.5` per edge whose predicate matches a cue, `+1.5` per
intermediate node whose type the question named, `−0.6` per hop.

So *"who fixed the bug in the service that Atlas depends on?"* seeds on `Atlas`,
targets `Person`, cues `{fixed_by, depends_on}`, and the winning path —

```
Atlas -[depends on]-> Ember <-[affects]- BUG-903 -[fixed by]-> Marcus Chen
```

— scores highly because two of its three edges match cues and it routes through
a `Bug`, which the question named. That path is the thing RAG cannot produce,
because no passage contains it.

Traverse undirected over a directed graph, but keep each edge's real direction
for rendering so a path can display `<-[affects]-` and stay honest.

`"within two hops"` is a separate neighbourhood query, and worth calling out:
proximity in a graph is a fact, proximity in an embedding space is a vibe. RAG
cannot even represent the question.

The corresponding weakness is just as real. Anything not expressible as a
declared relation is invisible, and procedural knowledge — *"what should I check
when the product feels slow?"* — has no edges at all.

---

## LLM wiki

### The idea

[Karpathy's framing](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
is that you compile knowledge the way you compile code: an LLM reads immutable
raw sources and maintains a derived, human-readable wiki. Three operations —
**ingest**, **query**, **lint**.

The economic difference from RAG is the whole point:

| | build time | query time |
|---|---|---|
| RAG | embed chunks (seconds) | retrieve + **synthesise, every time** |
| Wiki | **synthesise, once** (minutes) | navigate + read |

RAG re-derives the same cross-document synthesis on every identical query,
forever. The wiki does it once and writes it down. That's why it wins on *"how
does all this fit together"* — and why it goes stale in a way RAG can't, since a
stale page is still fluent and well-cited. That's what lint exists for.

### Layers

```
data/raw/*.txt   immutable plain text. The compiler reads, never writes.
wiki/pages/*.md  generated markdown. The compiler owns these entirely.
wiki/index.md    content catalogue: every page, one-line summary, by category.
wiki/log.md      append-only: INGEST | COMPILE | QUERY | LINT, timestamped.
```

Karpathy's version has a third "schema layer" — a document telling the LLM how
the wiki is organised and which conventions to follow. Here that lives in the
compiler's own module docstring plus the page format below, so the conventions
travel with the code that enforces them.

### Page format

Fixed, because the query layer parses it:

```markdown
---
title: Atlas
page_type: entity
entity_type: Service
sources: [svc-atlas, run-atlas, arch-overview, inc-2041]
updated: 2026-07-31
---

# Atlas

> One-sentence summary. This is what index.md quotes.

## What it is
...prose...

## Relationships
- depends on [[Ember]]
- owned by [[Platform Team]]

## History
...prose referencing [[INC-2041]]...

## Related
[[Ember]] · [[Platform Team]] · [[Northwind Cloud]]

## Sources
- svc-atlas
- run-atlas
```

`[[Wiki Link]]` targets must exactly match page titles — lint uses them to find
orphans and broken links.

Relationship bullets are generated **from the knowledge graph**. Sharing the
extraction layer is deliberate: it keeps the comparison about representation and
access pattern rather than about who parsed better.

**Topic pages** are what beat RAG — "Architecture Overview", "Incident History",
"Vendor Risk", "Who Owns What" — because their content exists in no single
source document. That's precisely where chunk retrieval fragments.

### Query: navigation, not similarity

```
1. scan index.md      match question against page TITLES and SUMMARIES only
2. follow [[links]]   one hop out from the matched pages
3. read whole pages   evidence is pages, not chunks
4. cite twice         the page, and the raw doc_ids that page was built from
```

Scoring against titles and summaries rather than bodies is what keeps this a
table-of-contents scan instead of a full-text search, and it's why the one-liners
the compiler writes actually matter. **This approach must not embed anything.**

Two-level provenance — answer → wiki page → source documents — is a real
advantage over both other approaches. Surface it prominently in the UI.

### Lint

The self-maintenance pass, and the operation neither other approach has any
answer to. A vector index cannot notice that two of its chunks contradict each
other.

| Check | Catches |
|---|---|
| `orphan_pages` | pages nothing links to |
| `broken_links` | `[[Targets]]` resolving to nothing — usually a casing bug |
| `contradictions` | two objects for a functional predicate (`owned_by`, `led_by`, `caused_by`, `supersedes`) |
| `stale_pages` | newest source older than `wiki.stale_after_days` |
| `superseded_refs` | citing Data Retention Policy **v2** without saying it's superseded |
| `coverage_gaps` | raw documents no page cites |
| `index_drift` | `index.md` and `wiki/pages/` disagreeing |

`superseded_refs` earns its keep. Confidently answering from a document that has
been replaced is the most dangerous failure mode any knowledge base has, and the
corpus contains a planted instance of it.

---

## Frontend

Vanilla JS, no build step, no CDN. It has to work offline.

**Compare tab.** Three structurally identical columns, then a scoreboard:
latency, evidence tokens, documents touched, derivation shown y/n, admitted
ignorance y/n. That last row is the honest one.

**Graph tab.** Hand-rolled force-directed layout on canvas — repulsion between
all node pairs, attraction along edges, light centering, velocity damping, stop
when kinetic energy drops below epsilon. Node colour from ontology entity types.

The payoff: when a multi-hop question is asked, the answering path lights up.
That single visual does more to explain the difference between these
architectures than any amount of prose.

**Wiki tab.** Page browser with clickable backlinks, plus a lint button.

**Sources tab.** All 39 raw documents, so every citation anywhere is traceable.

---

## The corpus

39 **plain-text** documents about a fictional company, Meridian Systems: five
services, four teams, nine people, four incidents, three bugs, five policies,
four runbooks, two vendors, three reference docs.

It's synthetic on purpose and reverse-engineered from the demo questions: facts
are deliberately scattered so that multi-hop questions are genuinely multi-hop.
`bug-903.txt` says BUG-903 affects Ember and was fixed by Marcus Chen;
`svc-atlas.txt` says Atlas depends on Ember. Three files, one three-hop question.

The RAG-favouring questions work the same way in reverse. The slow-response
runbook never uses the words "sluggish" or "unresponsive", so lexical search
whiffs and only embeddings find it.

### Why .txt and not markdown

Because markdown in this repo means *generated output*. The sources are
unstructured prose the way a real corpus is; the only `.md` files anything
writes are wiki pages. You can see the difference between the three approaches
in `ls` before you read a line of code.

It also removes a quiet advantage. Frontmatter would have handed every approach
a free `type` and `tags` field, and a knowledge graph that reads its edges out of
YAML isn't demonstrating extraction. What structure remains is a naming
convention (`svc-`, `inc-`, `bug-`, …) and a three-line human header — the kind
of thing real corpora do give you, and nothing more.

Note that the embedding model neither knows nor cares about any of this: it
receives a string. The format question is about honesty of the demo, not about
what the model can ingest.

### Using it

Treat it as a regression fixture with known-correct answers. If you swap in real
documents later, keep this set for the tests.
