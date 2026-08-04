/* kb-lab graph tab.

   The knowledge graph, drawn. Entity type is the only hue on this canvas; the
   answering path is lit by contrast and weight rather than by a ninth colour,
   which is also the reason it survives greyscale and colour blindness. Every
   edge on screen can be clicked back to the sentence it was extracted from --
   non-negotiable #8 is the whole reason this tab is worth building.

   Loaded after layout.js and app.js and borrowing their globals (ForceLayout,
   getJSON, openSource, el, tag, tagged). There is no module system here
   because there is no build step, and there is no build step because this has
   to run on a plane. */

const TAU = Math.PI * 2;
const CURVE = 26;      // pixels of bow per step away from a straight line
const HIT = 8;         // click slop, in pixels
const ROUTE_CAP = 8;   // a neighbourhood query can return more routes than a picture can hold

const graph = {
  loaded: false,
  layout: null,
  nodes: [],
  edges: [],
  byId: new Map(),
  lit: { nodes: new Set(), edges: new Set() },
  routes: [],
  asked: "",
  kg: null,
  hover: null,
  selection: null,
  drag: null,
  ink: null,
  frame: 0,
};

// --- boot -------------------------------------------------------------------

document.addEventListener("kb-lab:panel", (event) => {
  if (event.detail.name === "graph") openGraph();
});

document.addEventListener("kb-lab:answered", (event) => {
  lightPath(event.detail.question, event.detail.answers);
});

async function openGraph() {
  if (!graph.loaded) {
    try {
      const body = await getJSON("/api/graph");
      build(body);
      graph.loaded = true;
    } catch (error) {
      showFailure(error);
      return;
    }
  }
  fitCanvas();
  // The route strip is drawn here rather than when the answer arrived: its
  // entity names are controls that need the loaded graph behind them, and a
  // question can be asked long before this tab is ever opened.
  if (graph.kg) renderRoutes();
  inspect(graph.selection);
  requestDraw();
}

function build(body) {
  graph.nodes = body.nodes.map((node) => ({ id: node.id, type: node.type, degree: 0, r: 6 }));
  graph.byId = new Map(graph.nodes.map((node) => [node.id, node]));

  graph.edges = body.edges.map((edge) => ({
    ...edge,
    key: edgeKey(edge.source, edge.predicate, edge.target),
    a: graph.byId.get(edge.source),
    b: graph.byId.get(edge.target),
    bow: 0,
  }));

  // Two relations between one pair of entities are two edges -- BUG-903 is
  // both reported_by and fixed_by Marcus Chen -- so parallel edges get bowed
  // apart. Drawn on top of each other they would look like one edge, and the
  // second fact would silently vanish from the picture.
  const parallel = new Map();
  for (const edge of graph.edges) {
    const pair = JSON.stringify([edge.source, edge.target].sort());
    const group = parallel.get(pair) || [];
    group.push(edge);
    parallel.set(pair, group);
  }
  for (const group of parallel.values()) {
    group.forEach((edge, index) => {
      edge.bow = (index - (group.length - 1) / 2) * CURVE * (group.length > 1 ? 1 : 0);
    });
  }

  for (const edge of graph.edges) {
    edge.a.degree += 1;
    edge.b.degree += 1;
  }
  for (const node of graph.nodes) node.r = 5 + Math.sqrt(node.degree) * 1.9;

  const links = graph.edges.filter((edge) => edge.a !== edge.b);
  graph.layout = new ForceLayout(graph.nodes, links);

  renderLegend();
  el("graph-count").textContent =
    `${graph.nodes.length} entities · ${graph.edges.length} relations`;
  canvas().setAttribute(
    "aria-label",
    `Knowledge graph of ${graph.nodes.length} entities and ${graph.edges.length} ` +
      "relations extracted from the corpus. The entity list beside it carries the same facts as text."
  );
  wireCanvas();
}

// JSON rather than a delimiter: entity names contain spaces, so a key built
// by joining them is a key two different triples could produce -- and a
// collision here is a path that lights the wrong edge.
const edgeKey = (source, predicate, target) => JSON.stringify([source, predicate, target]);
const canvas = () => el("graph-canvas");

// --- the canvas -------------------------------------------------------------

function fitCanvas() {
  const node = canvas();
  const width = node.clientWidth;
  const height = node.clientHeight;
  if (!width || !height) return; // the panel is still hidden; nothing to size to
  const dpr = window.devicePixelRatio || 1;
  node.width = Math.round(width * dpr);
  node.height = Math.round(height * dpr);
  node.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);

  if (!graph.layout.width) {
    relayout(width, height);
  } else {
    // Re-frame rather than re-simulate: a window resize should not rearrange a
    // graph the viewer has been reading, and may have dragged nodes around in.
    graph.layout.resize(width, height);
    graph.layout.frame();
  }
  requestDraw();
}

/* Settle the whole simulation off-screen, then frame the result. Both halves
   matter: an unsettled graph is spaghetti, and a settled one that was never
   framed sits in whichever corner the forces left it. */
function relayout(width, height) {
  graph.layout.seed(width, height);
  graph.layout.warm();
  graph.layout.frame();
}

new ResizeObserver(() => { if (graph.loaded) fitCanvas(); }).observe(document.documentElement);

// Both palettes live in the stylesheet, so a theme flip has to be re-read
// rather than recomputed: canvas wants colour strings, CSS owns the colours.
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  graph.ink = null;
  requestDraw();
});

function palette() {
  if (graph.ink) return graph.ink;
  const css = getComputedStyle(document.documentElement);
  const read = (name) => css.getPropertyValue(name).trim();
  const ink = {
    ink: read("--ink"),
    muted: read("--muted"),
    rule: read("--rule"),
    panel: read("--panel"),
    mono: read("--font-mono"),
    type: (type) => read(`--t-${type.toLowerCase()}`) || read("--muted"),
  };
  graph.ink = ink;
  return ink;
}

function requestDraw() {
  if (graph.frame) return;
  graph.frame = requestAnimationFrame(() => {
    graph.frame = 0;
    const moving = graph.layout && graph.layout.tick();
    draw();
    if (moving || graph.drag) requestDraw();
  });
}

// --- pointing at things -----------------------------------------------------

function wireCanvas() {
  const node = canvas();
  if (node.dataset.wired) return;
  node.dataset.wired = "yes";

  node.addEventListener("pointermove", (event) => {
    const point = where(event);
    if (graph.drag) {
      graph.drag.node.x = point.x;
      graph.drag.node.y = point.y;
      graph.drag.moved = true;
      graph.layout.reheat(0.25);
      requestDraw();
      return;
    }
    const found = pick(point);
    if (found?.node !== graph.hover?.node || found?.edge !== graph.hover?.edge) {
      graph.hover = found;
      node.style.cursor = found ? "pointer" : "default";
      requestDraw();
    }
  });

  node.addEventListener("pointerdown", (event) => {
    const found = pick(where(event));
    if (found?.node) {
      node.setPointerCapture(event.pointerId);
      found.node.pinned = true;
      graph.drag = { node: found.node, moved: false };
    }
  });

  node.addEventListener("pointerup", (event) => {
    const dragged = graph.drag;
    graph.drag = null;
    // A node dragged somewhere stays there; the viewer put it there on
    // purpose, and the simulation snatching it back is infuriating.
    if (dragged?.moved) return;
    if (dragged) dragged.node.pinned = false;
    select(pick(where(event)));
  });

  node.addEventListener("pointerleave", () => {
    graph.hover = null;
    requestDraw();
  });

  el("graph-reset").addEventListener("click", () => {
    for (const point of graph.nodes) point.pinned = false;
    relayout(canvas().clientWidth, canvas().clientHeight);
    requestDraw();
  });
}

function where(event) {
  const box = canvas().getBoundingClientRect();
  return { x: event.clientX - box.left, y: event.clientY - box.top };
}

/* Quadratic bezier bowed by `edge.bow`, so parallel relations stay apart and a
   self-relation still has something to draw. */
function curve(edge) {
  const { a, b } = edge;
  if (a === b) return { cx: a.x + 54, cy: a.y - 54 };
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const length = Math.hypot(dx, dy) || 1;
  return { cx: mx + (-dy / length) * edge.bow, cy: my + (dx / length) * edge.bow };
}

/* Nodes before edges: a node sits on top of every edge that touches it, so
   clicking one and getting the edge underneath would be a lie about z-order. */
function pick(point) {
  for (const node of graph.nodes) {
    if (Math.hypot(node.x - point.x, node.y - point.y) <= node.r + HIT) return { node };
  }
  for (const edge of graph.edges) {
    for (let t = 0; t <= 1.001; t += 1 / 16) {
      const on = at(edge, t);
      if (Math.hypot(on.x - point.x, on.y - point.y) <= HIT) return { edge };
    }
  }
  return null;
}

function incidentTo(found) {
  const edges = new Set();
  if (!found?.node) return edges;
  for (const edge of graph.edges) {
    if (edge.a === found.node || edge.b === found.node) edges.add(edge);
  }
  return edges;
}

function select(found) {
  graph.selection = found;
  inspect(found);
  requestDraw();
}
