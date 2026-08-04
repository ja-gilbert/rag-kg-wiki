# kb-lab

**Same corpus. Same question. Three architectures, side by side, in your browser.**

RAG, a knowledge graph, and a Karpathy-style LLM wiki all answer the same
question over the same 39 documents — and you get to see not just what each one
answered, but what evidence it found, how it found it, how long it took, and
where it quietly failed.

Runs entirely on `localhost`. No API keys required.

> **Status: all three approaches work, and the compare view is live.** Build the
> artefacts, run the server, and `localhost:8000` answers any of the ten demo
> questions in three columns with a scoreboard underneath. The graph, wiki and
> sources tabs are still to come. `docs/ARCHITECTURE.md` describes the design
> and the reasoning behind it.

---

## The three approaches

**RAG — vector retrieval.** Embed the question, cosine-search a chunk index,
hand the top *k* chunks to a generator. Cheap to build, fast to update, finds
things by meaning rather than keyword. Cannot answer anything that requires
joining facts across documents unless some passage happens to contain the join,
and has no way to distinguish "no good answer" from "a weak answer" — a
similarity score is never zero.

**Knowledge graph — typed relations, multi-hop traversal.** Extract entities and
relations from prose into a graph, then walk it. Answers questions like *"who
fixed the bug in the service that Atlas depends on?"* by producing a literal
path — `Atlas -[depends on]-> Ember <-[affects]- BUG-903 -[fixed by]-> Marcus
Chen` — with the source sentence behind every edge. Expensive to model, and
blind to anything not expressible as a declared relation.

**LLM wiki — compiled, cross-linked markdown.** The pattern
[Andrej Karpathy described](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f):
raw sources stay immutable, an LLM compiles them into a directory of
human-readable pages with `[[backlinks]]`, an `index.md` catalogue and an
append-only `log.md`. Queries navigate that structure instead of searching it.
Pays synthesis cost once at build time rather than on every query, which is why
it wins on broad "how does all this fit together" questions. Goes stale when
sources change — which is what the **lint** pass exists to catch, and which
neither of the other two approaches has any equivalent for.

---

## Questions worth asking

The repo ships with ten curated questions (`data/questions.yaml`), each chosen
because it makes one approach look good and another look bad **for a reason you
can explain**:

| Question | Winner | Why |
|---|---|---|
| Who fixed the bug in the service that Atlas depends on? | graph | Three hops. No passage contains the join, so retrieval has nothing to retrieve. |
| Which policy came out of the incident caused by the service that Beacon depends on? | graph | Same shape. Embeddings can't express a join. |
| Which people are within two hops of INC-2041? | graph | "Within two hops" isn't a property of text. RAG can't even represent the question. |
| A customer says the product feels sluggish. What should I check? | RAG | Procedural knowledge, so the graph has no edge to walk. Both retrievers find the runbook here; re-ask it as "pages take forever to appear" and only embeddings do. |
| How do the five services fit together and who owns what? | wiki | Answer spans eight documents. RAG returns five fragments each holding a fifth of it; the wiki already did the synthesis, offline. |
| What has repeatedly gone wrong with our vendors? | wiki | Two vendors, two incidents, one policy, one standing rule. Pre-compiled and backlinked. |
| **What was Meridian Systems' revenue last quarter?** | *none* | Not in the corpus. The honesty test — watch all three fail differently. |

That last one is the most useful thing in the repo, and it did not turn out the
way the design predicted. RAG answers it confidently: the top chunk scores
**0.466**, no lower than the questions RAG gets *right*, so no threshold
separates them. The wiki returns pages too — it matched on "Meridian" and
"system" and nothing else, which it says out loud. Only the graph finds no
entity to start from and declines.

Two of the three fail the honesty test. That is more useful than a tidy demo
where all three shrug politely, and it is why the threshold was left alone
rather than tuned until the numbers behaved.

---

## The corpus

39 plain-text documents about a fictional company, Meridian Systems: five
services, four teams, nine people, four incidents, three bugs, five policies,
four runbooks, two vendors, three reference docs.

It's synthetic on purpose, and reverse-engineered from the demo questions —
facts are deliberately scattered across files so that multi-hop questions are
genuinely multi-hop, rather than something you have to take on faith.

**The sources are `.txt` on purpose.** No frontmatter, no markup, no tags —
just a title line, a banner, a `Last updated:` line, and prose. Markdown in this
repo means *generated wiki output*, so you can see the difference between the
three approaches in a file listing before you read any code.

That also removes a quiet advantage: frontmatter would have handed every
approach a free `type` and `tags` field, and a knowledge graph that reads its
edges out of YAML isn't demonstrating extraction. Every graph edge is to be
pulled from prose using the patterns in `data/ontology.yaml`, and every edge
must remember the exact sentence it came from so it can be audited in the UI.

---

## Getting started

```powershell
.\setup.ps1          # Windows
```

```bash
./setup.sh           # macOS / Linux
```

That creates `.venv` and installs dependencies. Then build the artefacts once
and start the server:

```bash
python -m scripts.build_all     # embeds chunks, extracts triples, compiles the wiki
python -m app.main              # http://localhost:8000
```

`build_all` prints what each approach cost to build, which is half the
comparison:

```
rag    16.5s   1 index      (15.2s of it loading the embedding model)
kg      0.3s   64 edges
wiki    0.2s   36 pages
```

The first build downloads a ~80MB embedding model. After that everything runs
offline — set `embedding.backend: tfidf-svd` in `config.yaml` if you would
rather skip the download entirely.

### Without the browser

```bash
python -m scripts.ask "Who fixed the bug in the service that Atlas depends on?"
python -m scripts.ask --all     # every demo question, all three approaches
python -m scripts.lint          # check the compiled wiki; exits non-zero on errors
```

---

## What's here

```
config.yaml               every knob the three approaches disagree about
data/
  raw/*.txt               39 source documents  (immutable, never edited)
  ontology.yaml           entity types, aliases, relation patterns
  questions.yaml          the demo questions, and why each one matters
core/                     corpus loading, chunking, embeddings, vector store, LLM adapter
kgraph/                   relation extraction with provenance, graph traversal
wikigen/                  wiki compiler, page format, library, journal, lint
approaches/               rag.py, kg.py, wiki.py -- one Answer contract, one generator
app/                      FastAPI + vanilla-JS frontend, no build step
scripts/                  build_index, build_graph, build_wiki, build_all, ask, lint
tests/                    344 tests
docs/ARCHITECTURE.md      the design and the reasoning behind it
```

Generated, and safe to delete — `build_all` rebuilds them:

```
build/                    vectors, chunks, extracted triples
wiki/                     pages/*.md, index.md, log.md
```

## Still to come

The graph tab — a force-directed canvas where the answering path lights up when
you ask a multi-hop question — plus the wiki browser with a lint button, and a
sources tab listing all 39 raw documents. `docs/ARCHITECTURE.md` describes what
each of them is for.

## License

MIT
