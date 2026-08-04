/* The reading half of the graph tab: the entity index, the facts about
   whatever is selected, the type legend, and the route the graph walked.

   Split from graph.js and graph-draw.js because it is the part that has to
   work without the canvas. Everything the picture can show, this shows as
   text a keyboard can reach and a screen reader can read -- which for a demo
   whose whole claim is auditable provenance is not a courtesy, it is the
   feature.

   Loaded after graph.js, whose `graph` object and `select` it uses. */

// --- the inspector ----------------------------------------------------------

function inspect(found) {
  const panel = el("inspect");
  panel.replaceChildren();
  if (found?.node) return panel.append(...entityView(found.node));
  if (found?.edge) return panel.append(...relationView(found.edge));
  panel.append(...indexView());
}

/* The default view, and the keyboard route into a canvas that has none: every
   entity in the graph, grouped by type, each one a button that selects it. */
function indexView() {
  const parts = [tagged("h3", "eyebrow", "Every entity in the graph")];
  const list = document.createElement("ul");
  list.className = "index";
  for (const [type, names] of byType()) {
    const item = document.createElement("li");
    item.append(swatch(type), tagged("span", "index__type", type));
    const row = document.createElement("div");
    row.className = "index__names";
    for (const name of names) row.append(entityButton(name));
    item.append(row);
    list.append(item);
  }
  parts.push(list);
  return parts;
}

function entityView(node) {
  const facts = graph.edges.filter((edge) => edge.a === node || edge.b === node);
  const docs = new Set(facts.flatMap((edge) => edge.doc_ids));
  const parts = [
    back(),
    header(node.type, node.id),
    tagged(
      "p",
      "inspect__meta",
      `${facts.length} ${facts.length === 1 ? "fact" : "facts"} · ` +
        `${docs.size} ${docs.size === 1 ? "document" : "documents"}`
    ),
  ];
  const list = document.createElement("ol");
  list.className = "facts";
  for (const edge of facts) list.append(factItem(edge, node));
  parts.push(list);
  return parts;
}

function relationView(edge) {
  return [
    back(),
    header("Relation", ""),
    factItem(edge, null, { open: true }),
  ];
}

/* One renderer for a fact wherever it appears. The relation is always shown in
   its real direction with the entity in hand as plain text and the other end
   as a button -- never rephrased into an inverse, because the server sends one
   label and inventing the other one here would be asserting a claim no
   document made. */
function factItem(edge, node, options = {}) {
  const item = document.createElement("li");
  const line = document.createElement("p");
  line.className = "fact__line";
  line.append(
    edge.a === node ? tagged("span", "fact__self", edge.source) : entityButton(edge.source),
    tagged("span", "fact__label", `—${edge.label}→`),
    edge.b === node ? tagged("span", "fact__self", edge.target) : entityButton(edge.target)
  );
  item.append(line);

  // Non-negotiable #8: the sentence the edge was extracted from travels with
  // it, and the document id below opens the raw prose it came from.
  const quote = tagged("p", "fact__sentence", `“${edge.sentence}”`);
  if (!options.open) quote.classList.add("fact__sentence--tight");
  item.append(quote);

  const row = document.createElement("div");
  row.className = "block__docs";
  for (const docId of edge.doc_ids) {
    const button = tagged("button", "doc", docId);
    button.type = "button";
    button.title = `Open ${docId} as raw text`;
    button.addEventListener("click", () => openSource(docId));
    row.append(button);
  }
  item.append(row);
  return item;
}

function entityButton(name) {
  const button = tagged("button", "entity", name);
  button.type = "button";
  const node = graph.byId.get(name);
  if (node) {
    button.style.setProperty("--channel", `var(--t-${node.type.toLowerCase()})`);
    button.addEventListener("click", () => select({ node }));
  }
  return button;
}

function header(kind, name) {
  const box = document.createElement("div");
  box.append(tagged("p", "inspect__kind", kind));
  if (name) box.append(tagged("h3", "inspect__name", name));
  return box;
}

function back() {
  const button = tagged("button", "ghost", "← all entities");
  button.type = "button";
  button.addEventListener("click", () => select(null));
  return button;
}

function byType() {
  const grouped = new Map();
  for (const node of [...graph.nodes].sort((a, b) => a.id.localeCompare(b.id))) {
    grouped.set(node.type, [...(grouped.get(node.type) || []), node.id]);
  }
  // Commonest type first, then alphabetical: a stable order that is about the
  // corpus rather than about the order the ontology happens to list types in.
  return [...grouped.entries()].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
}

function swatch(type) {
  const dot = document.createElement("span");
  dot.className = "swatch";
  dot.setAttribute("aria-hidden", "true");
  dot.style.setProperty("--channel", `var(--t-${type.toLowerCase()})`);
  return dot;
}

function renderLegend() {
  const legend = el("legend");
  legend.replaceChildren();
  for (const [type, names] of byType()) {
    const item = document.createElement("li");
    item.append(swatch(type), document.createTextNode(`${type} ${names.length}`));
    legend.append(item);
  }
}

// --- the answering path -----------------------------------------------------

/* The payoff. A multi-hop answer is a route through this graph, and no amount
   of prose explains the difference between these three architectures as
   quickly as watching that route light up while the rest of the graph dims. */
function lightPath(question, answers) {
  graph.asked = question;
  graph.kg = answers.find((answer) => answer.approach === "kg");
  graph.lit = { nodes: new Set(), edges: new Set() };
  // Exactly the blocks the approach handed its generator, read by shape rather
  // than by which kind of query produced them: a path question shows its one
  // route, a neighbourhood question shows the routes to everything it found.
  graph.routes = (graph.kg?.evidence || []).filter((block) => block.hops?.length).slice(0, ROUTE_CAP);

  for (const block of graph.routes) {
    for (const name of block.nodes) graph.lit.nodes.add(name);
    for (const hop of block.hops) {
      // A hop walked against its edge names the ends in traversal order, so
      // the edge it crossed is the reverse of what the hop reads.
      const [source, target] = hop.forward ? [hop.source, hop.target] : [hop.target, hop.source];
      graph.lit.edges.add(edgeKey(source, hop.predicate, target));
    }
  }

  document.querySelector('.tab[data-panel="graph"]').dataset.lit = String(graph.routes.length > 0);
  if (graph.loaded) {
    renderRoutes();
    requestDraw();
  }
}

function renderRoutes() {
  const strip = el("lit");
  const list = el("lit-routes");
  list.replaceChildren();

  el("lit-question").textContent = graph.asked;

  // Non-negotiable #7: the graph finding nothing is a result worth showing,
  // and this tab is where "no path" is at its most legible -- the picture
  // simply stays dark.
  if (!graph.routes.length) {
    list.append(tagged("li", "lit__none", graph.kg?.note || "The graph found no route to light up."));
    strip.hidden = false;
    return;
  }

  for (const block of graph.routes) {
    const item = document.createElement("li");
    item.append(entityButton(block.nodes[0]));
    block.hops.forEach((hop, index) => {
      item.append(arrowButton(hop), entityButton(block.nodes[index + 1]));
    });
    list.append(item);
  }
  strip.hidden = false;
}

function arrowButton(hop) {
  const [source, target] = hop.forward ? [hop.source, hop.target] : [hop.target, hop.source];
  const button = tagged(
    "button",
    "hop",
    hop.forward ? `—${hop.label}→` : `←${hop.label}—`
  );
  button.type = "button";
  button.title = hop.sentence;
  const edge = graph.edges.find((candidate) => candidate.key === edgeKey(source, hop.predicate, target));
  if (edge) button.addEventListener("click", () => select({ edge }));
  return button;
}
