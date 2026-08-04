/* kb-lab wiki tab.

   The compiled wiki, browsed the way the approach browses it: a catalogue that
   works like index.md, pages whose `[[links]]` are the only navigation, and
   the backlinks that come with them. No search box and no scoring on screen --
   non-negotiable #4 says this approach must not quietly become vector search
   over pages, and a tab that offered a similarity slider would be teaching the
   opposite of what it is here to teach.

   The lint button is the third of the approach's three operations. It reads,
   it never writes: /api/lint deliberately skips the log.md append, so a button
   somebody leans on cannot fill an append-only journal.

   Loaded after sources.js, whose document catalogue it borrows. */

const wiki = { pages: [], chosen: null, read: null, asked: "" };

document.addEventListener("kb-lab:panel", (event) => {
  if (event.detail.name === "wiki") openWiki();
});

document.addEventListener("kb-lab:answered", (event) => {
  wiki.asked = event.detail.question;
  wiki.read = event.detail.answers.find((answer) => answer.approach === "wiki") || null;
  if (wiki.pages.length) renderTrail();
});

async function openWiki() {
  try {
    if (!wiki.pages.length) {
      wiki.pages = (await getJSON("/api/wiki")).pages;
      renderCatalogue();
    }
    if (wiki.read) renderTrail();
    openPage(wiki.chosen || wiki.pages[0]?.title);
  } catch (error) {
    showFailure(error);
  }
}

// --- the catalogue ----------------------------------------------------------

/* index.md, as a panel. Topic pages are listed under their own heading rather
   than mixed in: they are the ones that exist because the compiler decided a
   subject needed a page, not because an entity did, and that difference is
   most of what separates this approach from the other two. */
function renderCatalogue() {
  const panel = el("catalogue");
  panel.replaceChildren(tagged("h3", "eyebrow", `${wiki.pages.length} pages, compiled from 39 documents`));

  const list = document.createElement("ul");
  list.className = "index";
  for (const [category, group] of grouped(wiki.pages, (page) => page.category)) {
    const item = document.createElement("li");
    item.append(tagged("span", "index__type", `${category} ${group.length}`));
    const names = document.createElement("div");
    names.className = "index__names";
    for (const page of group) names.append(pageButton(page.title, page.summary));
    item.append(names);
    list.append(item);
  }
  panel.append(list);
}

function pageButton(title, hint) {
  const button = tagged("button", "entity", title);
  button.type = "button";
  if (hint) button.title = hint;
  button.addEventListener("click", () => openPage(title));
  return button;
}

// --- a page -----------------------------------------------------------------

async function openPage(title) {
  if (!title) return;
  wiki.chosen = title;
  try {
    const page = await getJSON(`/api/wiki/${encodeURIComponent(title)}`);
    const stage = el("page");
    stage.replaceChildren(
      tagged("p", "stage__kind", `${page.page_type}${page.entity_type ? ` · ${page.entity_type}` : ""} · updated ${page.updated.slice(0, 10)}`),
      tagged("h2", "stage__title", page.title),
      tagged("p", "page__summary", page.summary)
    );
    for (const section of page.sections) stage.append(renderSection(section, page));
    stage.append(links("Backlinks — pages that point here", page.backlinks));
    stage.append(sourceRow(page.sources));

    for (const button of el("catalogue").querySelectorAll(".entity")) {
      button.dataset.current = String(button.textContent === page.title);
    }
    stage.scrollTop = 0;
  } catch (error) {
    showFailure(error);
  }
}

function renderSection(section, page) {
  const box = document.createElement("section");
  box.className = "page__section";
  box.append(tagged("h3", "eyebrow", section.heading));
  // Line by line, because the compiler writes bullets and this keeps them
  // bullets without handing a markdown parser a string it did not write.
  for (const line of section.body.split("\n")) {
    if (!line.trim()) continue;
    box.append(renderLine(line, page));
  }
  return box;
}

/* `[[Links]]` become buttons, and a trailing `(doc-id)` becomes a document
   button -- but only when the id is really a document. Everything else stays
   text. A citation you cannot follow is decoration, and a parenthesis dressed
   up as a citation is worse than decoration. */
function renderLine(line, page) {
  const row = document.createElement("p");
  row.className = line.startsWith("- ") ? "page__line page__line--bullet" : "page__line";
  const text = line.startsWith("- ") ? line.slice(2) : line;

  for (const part of text.split(/(\[\[[^\]]+\]\]|\([a-z0-9-]+\))/g)) {
    const link = part.match(/^\[\[([^\]]+)\]\]$/);
    if (link) {
      row.append(pageButton(link[1], wiki.pages.find((p) => p.title === link[1])?.summary));
      continue;
    }
    const cite = part.match(/^\(([a-z0-9-]+)\)$/);
    if (cite && page.sources.includes(cite[1])) {
      row.append(sourceButton(cite[1]));
      continue;
    }
    row.append(document.createTextNode(part));
  }
  return row;
}

function links(heading, titles) {
  const box = document.createElement("section");
  box.className = "page__section";
  box.append(tagged("h3", "eyebrow", heading));
  if (!titles.length) {
    box.append(tagged("p", "page__none", "Nothing links here yet."));
    return box;
  }
  const row = document.createElement("div");
  row.className = "index__names";
  for (const title of titles) row.append(pageButton(title));
  box.append(row);
  return box;
}

/* Non-negotiable #8: a page states which raw documents it was compiled from,
   and each one opens as unmodified text. */
function sourceRow(ids) {
  const box = document.createElement("section");
  box.className = "page__section";
  box.append(tagged("h3", "eyebrow", `Compiled from ${ids.length} raw ${ids.length === 1 ? "document" : "documents"}`));
  const row = document.createElement("div");
  row.className = "block__docs";
  for (const id of ids) row.append(sourceButton(id));
  box.append(row);
  return box;
}

function sourceButton(docId) {
  const button = tagged("button", "doc", docId);
  button.type = "button";
  button.title = `Open ${docId} as raw text`;
  button.addEventListener("click", () => openSource(docId));
  return button;
}

// --- what the approach read -------------------------------------------------

/* The wiki's answer to the graph tab's route: which pages the catalogue
   matched, and which it then reached by following [[links]]. Those are two
   different mechanisms and the whole approach rests on the second one, so they
   are shown apart rather than as one list of pages. */
function renderTrail() {
  const trail = el("read-trail");
  trail.replaceChildren();
  el("read-question").textContent = wiki.asked;

  const detail = wiki.read?.detail || {};
  const matched = detail.matched || [];
  const followed = detail.followed || [];

  if (!matched.length) {
    trail.append(tagged("p", "lit__none", wiki.read?.note || "The catalogue matched no page."));
    el("read").hidden = false;
    return;
  }

  trail.append(step("Matched on the catalogue", matched.map((page) => ({
    title: page.title,
    hint: `score ${page.score} · matched ${page.matched_on.join(", ")}`,
  }))));
  trail.append(step("Then followed [[links]] to", followed.map((title) => ({ title }))));

  // `pages_read` is the list of pages actually opened, not a count -- the cap
  // is what stops a well-linked wiki from being read end to end for one
  // question, and saying which pages made the cut is the honest version.
  const read = detail.pages_read || [];
  trail.append(
    tagged(
      "p",
      "read__count",
      `${read.length} of those were read in full, the most this configuration allows (max_pages ${detail.max_pages}).`
    )
  );
  el("read").hidden = false;
}

function step(heading, entries) {
  const box = document.createElement("div");
  box.className = "read__step";
  box.append(tagged("span", "read__label", heading));
  const row = document.createElement("div");
  row.className = "index__names";
  if (!entries.length) row.append(tagged("span", "page__none", "nothing"));
  for (const entry of entries) row.append(pageButton(entry.title, entry.hint));
  box.append(row);
  return box;
}

// --- lint -------------------------------------------------------------------

el("lint-run").addEventListener("click", runLint);

async function runLint() {
  const button = el("lint-run");
  button.disabled = true;
  button.textContent = "Linting";
  try {
    const report = await getJSON("/api/lint");
    renderLint(report);
  } catch (error) {
    showFailure(error);
  } finally {
    button.disabled = false;
    button.textContent = "Run lint";
  }
}

/* Grouped by check, because seven checks producing one finding each and one
   check producing seven are different situations and a flat list reads them
   the same. Findings are not errors to be hidden -- non-negotiable #7 -- so a
   clean run says so as plainly as a dirty one. */
function renderLint(report) {
  el("lint-verdict").textContent =
    `${report.page_count} pages · ${report.findings.length} findings · ` +
    `${Math.round(report.seconds * 1000)}ms · ${report.ok ? "no errors" : "errors found"}`;
  el("lint-verdict").dataset.ok = String(report.ok);

  const box = el("lint-findings");
  box.replaceChildren();
  if (!report.findings.length) {
    box.append(tagged("p", "lint__clean", "Every check passed. The wiki is coherent with its sources."));
    return;
  }
  for (const [check, group] of grouped(report.findings, (finding) => finding.check)) {
    const item = document.createElement("div");
    item.className = "lint__check";
    item.dataset.severity = group[0].severity;
    item.append(tagged("p", "lint__name", `${check} · ${group[0].severity} · ${group.length}`));
    const list = document.createElement("ul");
    for (const finding of group) {
      const row = document.createElement("li");
      row.append(tagged("span", "lint__subject", finding.subject), document.createTextNode(finding.detail));
      list.append(row);
    }
    item.append(list);
    box.append(item);
  }
}
