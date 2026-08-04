/* Painting the knowledge graph onto the canvas.

   Split from graph.js because none of this decides anything: it reads the
   node positions the layout produced and the sets graph.js maintains, and
   turns them into pixels. Nothing in here mutates the graph.

   Loaded after graph.js, whose `graph` object it reads throughout. */

// --- drawing ----------------------------------------------------------------

/* Four passes rather than one loop over everything, because the layers have to
   come out in this order: no disc may cover an edge it is not attached to, and
   no disc may cover a name. Drawing each node's label next to its own circle
   -- the obvious way -- means every label is at the mercy of whichever node
   happens to be drawn after it. */
function draw() {
  const node = canvas();
  const ctx = node.getContext("2d");
  const width = node.clientWidth;
  const height = node.clientHeight;
  if (!width || !height) return;
  const ink = palette();
  ctx.clearRect(0, 0, width, height);
  ctx.font = `11px ${ink.mono}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const lighting = graph.lit.nodes.size > 0;
  const near = incidentTo(graph.hover);
  const taken = [];
  const captioned = [];

  for (const edge of graph.edges) {
    const on = graph.lit.edges.has(edge.key);
    const picked = graph.selection?.edge === edge || graph.hover?.edge === edge || near.has(edge);
    ctx.globalAlpha = lighting ? (on ? 1 : 0.09) : picked ? 1 : 0.45;
    ctx.strokeStyle = on || picked ? ink.ink : ink.rule;
    ctx.lineWidth = on ? 2.4 : picked ? 1.7 : 1;
    strokeEdge(ctx, edge);
    arrowhead(ctx, edge, ctx.strokeStyle);
    if (on || picked) captioned.push({ edge });
  }

  for (const point of graph.nodes) {
    const on = !lighting || graph.lit.nodes.has(point.id);
    const picked = graph.selection?.node === point || graph.hover?.node === point;
    const colour = ink.type(point.type);
    ctx.globalAlpha = on ? 1 : 0.13;
    ctx.beginPath();
    ctx.arc(point.x, point.y, point.r, 0, TAU);
    ctx.fillStyle = ink.panel;
    ctx.fill();
    ctx.globalAlpha = on ? 0.24 : 0.05;
    ctx.fillStyle = colour;
    ctx.fill();
    ctx.globalAlpha = on ? 1 : 0.13;
    ctx.lineWidth = picked ? 3 : 1.7;
    ctx.strokeStyle = colour;
    ctx.stroke();
  }

  // Names before relations. Both are worth reading, but a relation label that
  // will not fit is still spelled out in the route strip above the canvas and
  // in the inspector beside it, where an unlabelled node is just a circle.
  const insisted = [];
  const rest = [];
  for (const point of ranked(lighting)) (shouted(point, lighting) ? insisted : rest).push(point);

  for (const point of insisted) {
    ctx.globalAlpha = 1;
    place(ctx, point, ink.ink, ink, taken, true);
  }
  for (const { edge } of captioned) {
    ctx.globalAlpha = 1;
    label(ctx, edge, ink, taken);
  }
  for (const point of rest) {
    ctx.globalAlpha = lighting ? 0.16 : 1;
    place(ctx, point, lighting ? ink.muted : ink.ink, ink, taken, false);
  }
  ctx.globalAlpha = 1;
}

const shouted = (point, lighting) =>
  (lighting && graph.lit.nodes.has(point.id)) ||
  graph.selection?.node === point ||
  graph.hover?.node === point;

/* Which names get to stay when they cannot all fit. Anything on the answering
   path first, then whatever is being pointed at, then the best-connected
   entities -- a hub with eight relations is more use as a landmark than a leaf
   with one, and the leaf is one hover away from saying its name anyway. */
function ranked(lighting) {
  const score = (point) => (shouted(point, lighting) ? 100 : 0) + point.degree;
  return [...graph.nodes].sort((a, b) => score(b) - score(a));
}

/* A name below its node, or above, or to one side -- whichever is free. Names
   that fit nowhere are dropped rather than overlapped, except on the lit path,
   where the picture is answering a question and every name on it has to be
   readable. Those fall back to whichever position treads on the least, which
   is what keeps two adjacent lit nodes from stacking their names in the same
   spot -- exactly the case a path through a tight cluster produces. */
function place(ctx, point, colour, ink, taken, insist) {
  const half = ctx.measureText(point.id).width / 2;
  const spots = [
    [point.x, point.y + point.r + 11],
    [point.x, point.y - point.r - 11],
    [point.x + point.r + half + 6, point.y],
    [point.x - point.r - half - 6, point.y],
  ];
  for (const [x, y] of spots) {
    if (caption(ctx, point.id, x, y, colour, ink, taken)) return;
  }
  if (!insist) return;
  let best = spots[0];
  let least = Infinity;
  for (const [x, y] of spots) {
    const cost = crowding(boxOf(ctx, point.id, x, y), taken);
    if (cost < least) { least = cost; best = [x, y]; }
  }
  caption(ctx, point.id, best[0], best[1], colour, ink, taken, true);
}

function strokeEdge(ctx, edge) {
  const { cx, cy } = curve(edge);
  ctx.beginPath();
  ctx.moveTo(edge.a.x, edge.a.y);
  ctx.quadraticCurveTo(cx, cy, edge.b.x, edge.b.y);
  ctx.stroke();
}

function at(edge, t) {
  const { cx, cy } = curve(edge);
  const u = 1 - t;
  return {
    x: u * u * edge.a.x + 2 * u * t * cx + t * t * edge.b.x,
    y: u * u * edge.a.y + 2 * u * t * cy + t * t * edge.b.y,
  };
}

/* Direction matters: `INC-2088 caused_by Cinder` read backwards is a different
   claim, and a path that walked it backwards says so. */
function arrowhead(ctx, edge, colour) {
  const tip = at(edge, 0.94);
  const back = at(edge, 0.82);
  const angle = Math.atan2(tip.y - back.y, tip.x - back.x);
  const x = edge.b.x - Math.cos(angle) * (edge.b.r + 2.5);
  const y = edge.b.y - Math.sin(angle) * (edge.b.r + 2.5);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - Math.cos(angle - 0.42) * 8, y - Math.sin(angle - 0.42) * 8);
  ctx.lineTo(x - Math.cos(angle + 0.42) * 8, y - Math.sin(angle + 0.42) * 8);
  ctx.closePath();
  ctx.fillStyle = colour;
  ctx.fill();
}

/* Relation labels sit beside their edge rather than on it, tried at a few
   points along it and on either side, and dropped rather than laid over
   anything already drawn. Short edges go unlabelled.

   All of it is about the lit path, where four nodes, their names and three
   relation labels can land in one corner of the picture at once. A relation is
   the cheapest thing to lose there: the route strip above the canvas spells
   every one of them out in order, and an entity name that has been covered up
   leaves a circle nobody can identify. */
function label(ctx, edge, ink, taken) {
  if (Math.hypot(edge.b.x - edge.a.x, edge.b.y - edge.a.y) < 64) return;
  for (const [t, side] of [[0.5, 1], [0.5, -1], [0.34, 1], [0.66, 1], [0.34, -1], [0.66, -1]]) {
    const on = at(edge, t);
    const ahead = at(edge, t + 0.06);
    const dx = ahead.x - on.x;
    const dy = ahead.y - on.y;
    const length = Math.hypot(dx, dy) || 1;
    const x = on.x + ((-dy / length) * 9 * side);
    const y = on.y + ((dx / length) * 9 * side);
    if (caption(ctx, edge.label, x, y, ink.ink, ink, taken)) return;
  }
}

const boxOf = (ctx, text, x, y) => {
  const half = ctx.measureText(text).width / 2 + 3;
  return { x1: x - half, y1: y - 7, x2: x + half, y2: y + 7 };
};

const hits = (box, other) =>
  box.x1 < other.x2 && box.x2 > other.x1 && box.y1 < other.y2 && box.y2 > other.y1;

/* How much of `box` is already spoken for, in square pixels. */
const crowding = (box, taken) =>
  taken.reduce((total, other) => {
    if (!hits(box, other)) return total;
    return (
      total +
      (Math.min(box.x2, other.x2) - Math.max(box.x1, other.x1)) *
        (Math.min(box.y2, other.y2) - Math.max(box.y1, other.y1))
    );
  }, 0);

/* Text with the paper punched out behind it, claiming the rectangle it used.
   Returns false without drawing if that rectangle is already spoken for;
   `insist` is how a caller says this one goes down regardless. Labels sit on
   top of edges, and an unbacked or doubled-up one is unreadable exactly where
   the graph is busiest. */
function caption(ctx, text, x, y, colour, ink, taken, insist = false) {
  const box = boxOf(ctx, text, x, y);
  if (!insist && taken.some((other) => hits(box, other))) return false;
  taken.push(box);
  const alpha = ctx.globalAlpha;
  ctx.globalAlpha = alpha * 0.85;
  ctx.fillStyle = ink.panel;
  ctx.fillRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
  ctx.globalAlpha = alpha;
  ctx.fillStyle = colour;
  ctx.fillText(text, x, y);
  return true;
}
