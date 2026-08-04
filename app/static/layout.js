/* Force-directed layout, hand-rolled.

   32 nodes and 64 edges means the naive O(n^2) repulsion is about five hundred
   pairs a tick -- nothing -- and a hand-rolled simulation is far more legible
   than a library this repo would then have to ship offline.

   Kept apart from graph.js because none of this knows what a canvas is: it
   moves numbered points around a rectangle, and something else decides what
   they look like. */

/* Deterministic hash in [0, 1). The demo must draw the same picture on every
   reload -- a graph that rearranges itself between two runs of the same
   question is one the audience cannot compare against what they just saw. */
function hashUnit(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967296;
}

class ForceLayout {
  constructor(nodes, links, options = {}) {
    this.nodes = nodes;
    this.links = links;
    this.rest = options.rest ?? 96;
    this.repel = options.repel ?? 7200;
    this.spring = options.spring ?? 0.02;
    this.centre = options.centre ?? 0.006;
    this.damp = options.damp ?? 0.82;
    this.decay = options.decay ?? 0.988;
    this.floor = options.floor ?? 16; // never divide by a distance near zero
    this.alpha = 0;
    this.width = 0;
    this.height = 0;
  }

  /* Nodes start on a ring, in the order given, at a radius jittered by their
     own name. A ring beats random placement because no two nodes begin on top
     of each other, so the first few ticks untangle the graph instead of
     exploding it. */
  seed(width, height) {
    this.resize(width, height);
    const radius = Math.min(width, height) * 0.34;
    this.nodes.forEach((node, index) => {
      const angle = (index / this.nodes.length) * Math.PI * 2;
      const spread = radius * (0.7 + 0.55 * hashUnit(node.id));
      node.x = this.cx + Math.cos(angle) * spread;
      node.y = this.cy + Math.sin(angle) * spread;
      node.vx = 0;
      node.vy = 0;
      node.pinned = false;
    });
    this.alpha = 1;
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this.cx = width / 2;
    this.cy = height / 2;
  }

  /* Move and scale the settled graph to fill the canvas.

     Doing this rather than fencing the simulation inside the viewport, which
     was the first attempt: a boundary clamp does not make a layout fit, it
     makes the nodes that would have gone outside pile up against the wall,
     and on this corpus that squashed a quarter of the graph into the bottom
     edge with its labels hanging off the canvas.

     `rest` and `repel` are rescaled with the positions. The force law is only
     scale-covariant if the spring length grows as k and the repulsion as k^3,
     and without that the next tick would spend itself undoing the framing. */
  frame(pad = {}) {
    const left = pad.x ?? 96; // long entity names are drawn centred under the node
    const top = pad.top ?? 26;
    const bottom = pad.bottom ?? 36;
    const xs = this.nodes.map((node) => node.x);
    const ys = this.nodes.map((node) => node.y);
    const spanX = Math.max(...xs) - Math.min(...xs);
    const spanY = Math.max(...ys) - Math.min(...ys);
    const k = Math.min(
      (this.width - left * 2) / Math.max(spanX, 1),
      (this.height - top - bottom) / Math.max(spanY, 1)
    );
    if (!Number.isFinite(k) || k <= 0) return;

    const ox = (this.width - spanX * k) / 2 - Math.min(...xs) * k;
    const oy = top + (this.height - top - bottom - spanY * k) / 2 - Math.min(...ys) * k;
    for (const node of this.nodes) {
      node.x = node.x * k + ox;
      node.y = node.y * k + oy;
      node.vx = 0;
      node.vy = 0;
    }
    this.rest *= k;
    this.repel *= k ** 3;
  }

  reheat(alpha = 0.45) {
    this.alpha = Math.max(this.alpha, alpha);
  }

  get settled() {
    return this.alpha < 0.006;
  }

  /* One integration step. Returns false once the simulation has cooled, which
     is the renderer's cue to stop asking for frames. */
  tick() {
    if (this.settled) return false;
    const { nodes } = this;

    /* A force layout settles into a roughly circular blob, and a circle in a
       2:1 frame wastes half the frame. Repulsion is therefore stronger along
       the frame's long axis, which stretches the whole picture toward the
       shape of the canvas. Applied to every pair equally, so it is a global
       preference rather than a distortion of any particular part -- and it
       beats stretching the finished layout, which would leave edges pointing
       one way conspicuously longer than edges pointing the other. */
    const squeeze = Math.sqrt(this.height / this.width) || 1;

    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let distance = Math.hypot(dx, dy);
        if (distance < 0.5) {
          // Two nodes exactly coincident have no direction to separate along,
          // so borrow one from their names rather than from Math.random --
          // determinism has to survive the degenerate case too.
          dx = hashUnit(a.id + b.id) - 0.5;
          dy = hashUnit(b.id + a.id) - 0.5;
          distance = Math.hypot(dx, dy) || 0.5;
        }
        const force = this.repel / Math.max(distance * distance, this.floor * this.floor);
        const ux = ((dx / distance) * force) / squeeze;
        const uy = (dy / distance) * force * squeeze;
        a.vx += ux; a.vy += uy;
        b.vx -= ux; b.vy -= uy;
      }
    }

    for (const link of this.links) {
      if (link.a === link.b) continue; // a self-relation has no length to correct
      const dx = link.b.x - link.a.x;
      const dy = link.b.y - link.a.y;
      const distance = Math.hypot(dx, dy) || 0.5;
      const force = (distance - this.rest) * this.spring;
      const ux = (dx / distance) * force;
      const uy = (dy / distance) * force;
      link.a.vx += ux; link.a.vy += uy;
      link.b.vx -= ux; link.b.vy -= uy;
    }

    for (const node of nodes) {
      if (node.pinned) { node.vx = 0; node.vy = 0; continue; }
      node.vx += (this.cx - node.x) * this.centre;
      node.vy += (this.cy - node.y) * this.centre;
      node.vx *= this.damp;
      node.vy *= this.damp;
      node.x += node.vx * this.alpha;
      node.y += node.vy * this.alpha;
    }

    this.alpha *= this.decay;
    return true;
  }

  /* Run the simulation to a standstill without drawing. Used to hand the
     viewer a graph that is already legible the moment the tab opens; watching
     thirty nodes find their places is charming exactly once, and this tab is
     opened after every question. */
  warm(limit = 900) {
    for (let i = 0; i < limit && this.tick(); i++);
  }
}
