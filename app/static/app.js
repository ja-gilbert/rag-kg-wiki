/* kb-lab compare tab.
   Vanilla, no build step, no CDN -- it has to run on a plane.

   Everything server-supplied goes in through textContent. Nothing here builds
   markup from a string, which matters more than usual: the text on screen is
   raw corpus prose, and the corpus is the one thing this repo promises not to
   sanitise or reshape. */

const el = (id) => document.getElementById(id);

// The scoreboard, in the reader's words rather than the envelope's field
// names. `kind` decides how a row is drawn, not which approach it belongs to:
// the rows are identical across columns for the same reason the columns are.
const METRICS = [
  { key: "latency_ms", label: "Time to answer", unit: "ms", kind: "bar" },
  { key: "evidence_tokens", label: "Evidence handed to the generator", unit: "tok", kind: "bar" },
  { key: "documents_touched", label: "Source documents used", unit: "", kind: "bar" },
  { key: "derivation_shown", label: "Showed its work", kind: "mark" },
  { key: "admitted_ignorance", label: "Said what it did not know", kind: "mark" },
];

const state = { asking: false };

// --- boot -------------------------------------------------------------------

async function boot() {
  const status = await getJSON("/api/status");
  renderConfig(status);
  renderBanner(status);

  const { questions } = await getJSON("/api/questions");
  renderDemos(questions, status.ready);

  el("ask").addEventListener("submit", (event) => {
    event.preventDefault();
    ask(el("question").value);
  });

  if (!status.ready) {
    el("question").disabled = true;
    el("ask-button").disabled = true;
  }
}

function renderConfig(status) {
  const shown = {
    embedding: status.config.embedding,
    chunking: status.config.chunking,
    llm: status.config.llm,
    retrieval: `${status.config.rag_mode}/${status.config.rag_top_k}`,
    docs: status.documents,
    pages: status.pages,
    edges: status.edges,
  };
  const list = el("config");
  for (const [name, value] of Object.entries(shown)) {
    const row = document.createElement("div");
    row.append(tag("dt", name), tag("dd", String(value)));
    list.append(row);
  }
}

function renderBanner(status) {
  if (status.ready) return;
  const banner = el("banner");
  banner.append(
    document.createTextNode("The build artefacts are not there yet. Run "),
    tag("code", status.build_command),
    document.createTextNode(", then reload this page. The sources below the fold still work.")
  );
  banner.hidden = false;
}

function renderDemos(questions, ready) {
  const list = el("demos");
  for (const row of questions) {
    const button = tag("button", "");
    button.type = "button";
    button.className = "demo";
    button.disabled = !ready;
    button.title = row.why ? row.why.trim() : "";
    button.style.setProperty("--channel", `var(--${row.expect})`);

    const dot = document.createElement("span");
    dot.className = "demo__expect";
    dot.setAttribute("aria-hidden", "true");
    button.append(dot, document.createTextNode(row.q));
    button.addEventListener("click", () => ask(row.q));

    const item = document.createElement("li");
    item.append(button);
    list.append(item);
  }
}

// --- asking -----------------------------------------------------------------

async function ask(question) {
  const asked = question.trim();
  if (!asked || state.asking) return;

  state.asking = true;
  el("question").value = asked;
  el("ask-button").disabled = true;
  el("ask-button").textContent = "Asking";

  try {
    const body = await postJSON("/api/ask", { question: asked });
    el("empty").hidden = true;
    renderColumns(body.answers);
    renderBoard(body.scoreboard);
  } catch (error) {
    showFailure(error);
  } finally {
    state.asking = false;
    el("ask-button").disabled = false;
    el("ask-button").textContent = "Ask";
  }
}

function renderColumns(answers) {
  const columns = el("columns");
  columns.replaceChildren();
  for (const answer of answers) columns.append(renderColumn(answer));
}

function renderColumn(answer) {
  const node = el("column-template").content.cloneNode(true);
  const column = node.querySelector(".column");
  column.dataset.approach = answer.approach;

  q(node, ".column__label").textContent = answer.label;
  q(node, ".readout__ms").textContent = `${Math.round(answer.ms.total)}ms`;
  q(node, ".readout__tokens").textContent = `${answer.tokens_est} tok`;
  q(node, ".readout__dot").dataset.confident = String(answer.confident);
  q(node, ".readout__dot").title = answer.confident
    ? "This approach stands behind its answer"
    : "This approach flagged its own answer";

  // An approach with nothing to say puts its explanation in both fields. The
  // answer slot stays but stays empty -- "No answer." followed by the reason
  // reads as a result, where the same sentence twice reads as a bug.
  const echoed = answer.note && answer.answer.trim() === answer.note.trim();
  if (!echoed) renderAnswer(q(node, ".answer"), answer.answer, column);

  // Non-negotiable #7: an approach admitting it has nothing is a result, so
  // the note gets the same weight as the answer rather than being tucked away.
  if (answer.note) {
    const note = q(node, ".note");
    note.textContent = answer.note;
    note.hidden = false;
  }

  const steps = q(node, ".trace__steps");
  for (const step of answer.trace) steps.append(tag("li", step));

  q(node, ".evidence__count").textContent =
    answer.evidence.length === 1 ? "1 piece of evidence" : `${answer.evidence.length} pieces of evidence`;
  const blocks = q(node, ".evidence__blocks");
  answer.evidence.forEach((block, index) => blocks.append(renderBlock(block, index + 1)));

  return node;
}

/* The generator writes citations as [1], [2]. They are the only interactive
   thing inside generated prose, so they are parsed out and turned into buttons
   that open the evidence they point at -- a citation you cannot follow is
   decoration. */
function renderAnswer(target, text, column) {
  const parts = String(text || "").split(/(\[\d+\])/g);
  for (const part of parts) {
    const match = part.match(/^\[(\d+)\]$/);
    if (!match) {
      target.append(document.createTextNode(part));
      continue;
    }
    const number = Number(match[1]);
    const button = tag("button", part);
    button.type = "button";
    button.className = "cite";
    button.title = `Show evidence ${number}`;
    button.addEventListener("click", () => revealEvidence(column, number));
    target.append(button);
  }
}

function revealEvidence(column, number) {
  const details = column.querySelector(".evidence");
  details.open = true;
  const block = column.querySelectorAll(".evidence__blocks > li")[number - 1];
  if (block) block.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

/* One renderer for all three approaches. The blocks differ -- a chunk, a path,
   a page -- so the shared fields are read by shape rather than by approach
   name; anything else would put a per-approach branch in the one place the
   columns are supposed to be identical. */
function renderBlock(block, number) {
  const item = document.createElement("li");

  const head = document.createElement("div");
  head.className = "block__head";
  head.append(tagged("span", "block__number", `[${number}]`));

  const label = block.title || block.doc_title || block.entity || "";
  if (label) head.append(tagged("span", "block__title", label));
  if (block.score !== undefined) {
    head.append(tagged("span", "block__score", `score ${block.score}`));
  }
  item.append(head);

  item.append(tagged("p", "block__text", block.text || ""));

  const docs = block.doc_ids || (block.doc_id ? [block.doc_id] : []);
  if (docs.length) {
    const row = document.createElement("div");
    row.className = "block__docs";
    for (const docId of docs) {
      const button = tagged("button", "doc", docId);
      button.type = "button";
      button.title = `Open ${docId} as raw text`;
      button.addEventListener("click", () => openSource(docId));
      row.append(button);
    }
    item.append(row);
  }
  return item;
}

// --- the scoreboard ---------------------------------------------------------

/* Every row is drawn to one scale across all three columns. No axis, no
   gridlines, no winner marked: the numbers are printed, and which of them
   counts as better is exactly the question the demo refuses to answer for you. */
function renderBoard(rows) {
  const board = el("board-rows");
  board.replaceChildren();

  for (const metric of METRICS) {
    const line = document.createElement("div");
    line.className = "metric";
    line.append(tagged("div", "metric__label", metric.label));

    const values = rows.map((row) => row[metric.key]);
    const largest = Math.max(...values.map((v) => (typeof v === "number" ? v : 0)), 0);

    rows.forEach((row, index) => {
      const gauge = document.createElement("div");
      gauge.className = "gauge";
      gauge.dataset.approach = row.approach;
      gauge.title = row.label;

      if (metric.kind === "mark") {
        const on = Boolean(values[index]);
        const mark = tagged("span", "gauge__mark", on ? "● yes" : "○ no");
        mark.dataset.on = String(on);
        gauge.append(mark);
      } else {
        const track = document.createElement("div");
        track.className = "gauge__track";
        const fill = document.createElement("div");
        fill.className = "gauge__fill";
        fill.style.width = largest ? `${(values[index] / largest) * 100}%` : "0";
        fill.style.animationDelay = `${index * 70}ms`;
        track.append(fill);
        gauge.append(tagged("span", "gauge__value", format(values[index], metric.unit)), track);
      }
      line.append(gauge);
    });
    board.append(line);
  }
  el("board").hidden = false;
}

function format(value, unit) {
  const rounded = Number.isInteger(value) ? value : Math.round(value);
  return unit ? `${rounded}${unit}` : String(rounded);
}

// --- raw sources ------------------------------------------------------------

async function openSource(docId) {
  const dialog = el("source");
  try {
    const document_ = await getJSON(`/api/sources/${encodeURIComponent(docId)}`);
    q(dialog, ".source__title").textContent = document_.title;
    q(dialog, ".source__meta").textContent =
      `${document_.doc_id} · ${document_.doc_type}${document_.date ? ` · ${document_.date}` : ""}`;
    q(dialog, ".source__text").textContent = document_.text;
    dialog.showModal();
  } catch (error) {
    showFailure(error);
  }
}

// --- plumbing ---------------------------------------------------------------

function showFailure(error) {
  const banner = el("banner");
  banner.replaceChildren(document.createTextNode(String(error.message || error)));
  banner.hidden = false;
}

async function getJSON(url) {
  return unwrap(await fetch(url));
}

async function postJSON(url, payload) {
  return unwrap(
    await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

async function unwrap(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `${response.status} from the server`);
  return body;
}

const q = (root, selector) => root.querySelector(selector);

function tag(name, text) {
  const node = document.createElement(name);
  node.textContent = text;
  return node;
}

function tagged(name, className, text) {
  const node = tag(name, text);
  node.className = className;
  return node;
}

boot();
