from __future__ import annotations

from dataclasses import dataclass

from kgraph.extract import Triple

# What a knowledge graph is made of, and what it hands back when you walk it.
# Split from graph.py because these are value types concerned with provenance
# and rendering, while the graph itself is concerned with storage and
# traversal. The dependency runs one way: graph.py builds these, they know
# nothing about it.


@dataclass(frozen=True)
class Edge:
    """One relation between two entities, with every assertion of it.

    Two documents stating the same fact are two pieces of evidence for one
    edge, not two edges: kept apart, every path crossing the edge would be
    enumerated once per source with identical nodes and only the citation
    differing.
    """

    subject: str
    predicate: str
    object: str
    sources: tuple[Triple, ...]

    @property
    def sentence(self) -> str:
        return self.sources[0].sentence

    @property
    def doc_id(self) -> str:
        return self.sources[0].doc_id

    @property
    def doc_ids(self) -> tuple[str, ...]:
        # Order-preserving dedupe: one document can assert an edge twice.
        return tuple(dict.fromkeys(s.doc_id for s in self.sources))


@dataclass(frozen=True)
class Fact:
    """An edge phrased from the perspective of the entity it is a fact about.

    `BUG-903 affects Ember` is a fact about Ember too, but a panel headed
    "Ember" cannot list it in that form without reading backwards, so an
    inbound edge borrows the relation's inverse label.
    """

    entity: str
    other: str
    label: str
    outbound: bool
    edge: Edge

    @property
    def subject(self) -> str:
        return self.edge.subject

    @property
    def predicate(self) -> str:
        return self.edge.predicate

    @property
    def object(self) -> str:
        return self.edge.object

    @property
    def sentence(self) -> str:
        return self.edge.sentence

    @property
    def doc_id(self) -> str:
        return self.edge.doc_id

    def render(self) -> str:
        return f"{self.entity} {self.label} {self.other}"


@dataclass(frozen=True)
class Hop:
    """One edge, walked. `forward` is false when the walk ran against it.

    Stored rather than derived from `source == edge.subject` so that a
    self-referential edge -- a policy superseding itself, say -- degrades to a
    wrong arrow instead of an exception.
    """

    source: str
    target: str
    label: str
    forward: bool
    edge: Edge

    @property
    def sentence(self) -> str:
        return self.edge.sentence

    @property
    def doc_id(self) -> str:
        return self.edge.doc_id

    def render(self) -> str:
        return f"-[{self.label}]->" if self.forward else f"<-[{self.label}]-"


@dataclass(frozen=True)
class Path:
    nodes: tuple[str, ...]
    hops: tuple[Hop, ...]

    def render(self) -> str:
        """`Atlas -[depends on]-> Ember <-[affects]- BUG-903`.

        The arrow flips, never the label: a path that walked `BUG-903 affects
        Ember` backwards has to say so rather than invent the inverse claim.
        """
        parts = [self.nodes[0]]
        for hop in self.hops:
            parts += [hop.render(), hop.target]
        return " ".join(parts)

    def extend(self, hop: Hop) -> Path:
        return Path(self.nodes + (hop.target,), self.hops + (hop,))


@dataclass(frozen=True)
class Neighbour:
    name: str
    type: str
    distance: int
    # The shortest route found to it. Without this, "within two hops" is as
    # unfalsifiable as a similarity score.
    via: Path
