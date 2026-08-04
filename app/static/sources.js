/* kb-lab sources tab.

   All 39 raw documents, so that every citation anywhere in this app lands on
   actual text. Nothing here interprets anything: the corpus is the one thing
   the repo promises not to reshape, and this tab is where you go to check that
   it hasn't been.

   It also owns the document catalogue, because two tabs need it. wiki.js asks
   for it to tell a real citation from an ordinary parenthesis -- a `(...)`
   that is not a document id has no business looking like a link.

   Loaded after app.js, whose plumbing it borrows. */

const sources = { chosen: null, wired: false };
let catalogue = null;

/* One fetch, however many callers. Cached as the promise rather than the
   result so two tabs opening at once share the request instead of racing it. */
function documents() {
  if (!catalogue) catalogue = getJSON("/api/sources").then((body) => body.documents);
  return catalogue;
}

document.addEventListener("kb-lab:panel", (event) => {
  if (event.detail.name === "sources") openSources();
});

async function openSources() {
  if (sources.wired) return;
  try {
    renderDocIndex(await documents());
    sources.wired = true;
    showDocument(sources.chosen);
  } catch (error) {
    showFailure(error);
  }
}

/* Grouped by the filename prefix the corpus uses for its type -- svc-, inc-,
   bug- and the rest. That convention is all the structure a raw document has,
   and CLAUDE.md is emphatic that it stays that way: a corpus with metadata in
   it would make the knowledge-graph demo a demo of reading metadata. */
function renderDocIndex(rows) {
  const index = el("doc-index");
  index.replaceChildren(tagged("h3", "eyebrow", `All ${rows.length} raw documents`));

  const list = document.createElement("ul");
  list.className = "index";
  for (const [type, group] of grouped(rows, (row) => row.doc_type)) {
    const item = document.createElement("li");
    item.append(tagged("span", "index__type", `${type} ${group.length}`));
    const names = document.createElement("div");
    names.className = "index__names";
    for (const row of group) names.append(docButton(row));
    item.append(names);
    list.append(item);
  }
  index.append(list);
}

function docButton(row) {
  const button = tagged("button", "entity", row.doc_id);
  button.type = "button";
  button.title = row.title;
  button.addEventListener("click", () => showDocument(row.doc_id));
  return button;
}

async function showDocument(docId) {
  const rows = await documents();
  const chosen = rows.find((row) => row.doc_id === docId) || rows[0];
  if (!chosen) return;
  sources.chosen = chosen.doc_id;

  const stage = el("raw");
  try {
    const body = await getJSON(`/api/sources/${encodeURIComponent(chosen.doc_id)}`);
    stage.replaceChildren(
      tagged("p", "stage__kind", `${body.doc_type}${body.date ? ` · ${body.date}` : ""}`),
      tagged("h2", "stage__title", body.title),
      tagged("p", "stage__id", body.doc_id),
      // A <pre>, deliberately. These files are three header lines and then
      // prose, and re-flowing them into paragraphs here would quietly render
      // the corpus as something tidier than it is.
      tagged("pre", "raw__text", body.text)
    );
    for (const button of el("doc-index").querySelectorAll(".entity")) {
      button.dataset.current = String(button.textContent === chosen.doc_id);
    }
  } catch (error) {
    showFailure(error);
  }
}

/* Stable grouping for every index panel in the app: commonest kind first, then
   alphabetical, and the members sorted within each group. Shared with wiki.js,
   which groups pages by category and lint findings by check -- hence the
   name-or-id-or-subject sort key rather than one hard-coded field. */
function grouped(rows, key) {
  const groups = new Map();
  for (const row of rows) groups.set(key(row), [...(groups.get(key(row)) || []), row]);
  for (const group of groups.values()) group.sort((a, b) => sortKey(a).localeCompare(sortKey(b)));
  return [...groups.entries()].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
}

const sortKey = (row) => row.doc_id || row.title || row.subject || "";
