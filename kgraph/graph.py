from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator

import networkx as nx

from kgraph.elements import Edge, Fact, Hop, Neighbour, Path
from kgraph.extract import Triple
from kgraph.ontology import Ontology


class KnowledgeGraph:
    """A MultiDiGraph of extracted triples, walked as though undirected.

    Multi- because two relations between one pair of entities must coexist:
    `BUG-903 reported_by Marcus Chen` and `BUG-903 fixed_by Marcus Chen` are
    different facts, and a DiGraph would keep whichever arrived last.

    Traversal ignores direction -- you want to reach BUG-903 starting from the
    Ember it affects -- while every edge keeps its real direction for
    rendering, so a path can display `<-[affects]-` and stay honest.
    """

    def __init__(self, triples: list[Triple], ontology: Ontology) -> None:
        self.ontology = ontology
        self._graph = nx.MultiDiGraph()

        grouped: dict[tuple[str, str, str], list[Triple]] = {}
        for triple in triples:
            key = (triple.subject, triple.predicate, triple.object)
            grouped.setdefault(key, []).append(triple)

        for (subject, predicate, object_), sources in grouped.items():
            for name in (subject, object_):
                if name not in self._graph:
                    self._graph.add_node(name, type=ontology.entity(name).type)
            # Keyed on the predicate, so one relation between a pair is one
            # edge however many documents asserted it.
            self._graph.add_edge(
                subject,
                object_,
                key=predicate,
                edge=Edge(subject, predicate, object_, tuple(sources)),
            )

    # --- inspection ----------------------------------------------------------

    @property
    def nodes(self) -> list[str]:
        return list(self._graph.nodes)

    @property
    def edges(self) -> list[Edge]:
        return [data["edge"] for *_, data in self._graph.edges(data=True)]

    def type_of(self, entity: str) -> str:
        return self._graph.nodes[entity]["type"]

    def edge(self, subject: str, predicate: str, object_: str) -> Edge:
        return self._graph.edges[subject, object_, predicate]["edge"]

    def match(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> list[Edge]:
        """Triple-pattern matching; `None` in a slot is a wildcard."""
        return [
            e
            for e in self.edges
            if (subject is None or e.subject == subject)
            and (predicate is None or e.predicate == predicate)
            and (object is None or e.object == object)
        ]

    def facts_about(self, entity: str) -> list[Fact]:
        if entity not in self._graph:
            return []
        return [
            Fact(
                entity=entity,
                other=other,
                label=self._label(edge.predicate, forward),
                outbound=forward,
                edge=edge,
            )
            for other, edge, forward in self._steps(entity)
        ]

    # --- traversal -----------------------------------------------------------

    def neighbors_within(
        self, start: str, hops: int, types: Iterable[str] | None = None
    ) -> list[Neighbour]:
        """Every entity reachable in `hops` or fewer, optionally by type.

        The type filter applies to the results only. Filtering the walk itself
        would sever every route that crosses another type on its way -- and the
        route from an incident to a person runs through a service and a team.
        """
        if start not in self._graph:
            return []

        wanted = set(types) if types is not None else None
        routes: dict[str, Path] = {start: Path((start,), ())}
        queue = deque([start])

        while queue:
            node = queue.popleft()
            route = routes[node]
            if len(route.hops) >= hops:
                continue
            for target, edge, forward in self._steps(node):
                if target in routes:
                    continue
                routes[target] = route.extend(self._hop(node, target, edge, forward))
                queue.append(target)

        return sorted(
            (
                Neighbour(name, self.type_of(name), len(route.hops), route)
                for name, route in routes.items()
                if name != start and (wanted is None or self.type_of(name) in wanted)
            ),
            key=lambda n: (n.distance, n.name),
        )

    def paths_between(self, a: str, b: str, max_hops: int = 4) -> list[Path]:
        """All simple paths from `a` to `b`, shortest first.

        Breadth-first, so the queue is already in hop order and the caller can
        take the head as the best route. Returns empty rather than raising for
        an entity the graph has never heard of: "no path" is the honest answer
        to a question about something outside the corpus, and step 6 shows it.
        """
        if a not in self._graph or b not in self._graph or a == b:
            return []

        found: list[Path] = []
        queue = deque([Path((a,), ())])
        while queue:
            route = queue.popleft()
            if len(route.hops) >= max_hops:
                continue
            for target, edge, forward in self._steps(route.nodes[-1]):
                if target in route.nodes:
                    continue
                extended = route.extend(self._hop(route.nodes[-1], target, edge, forward))
                if target == b:
                    found.append(extended)
                else:
                    queue.append(extended)
        return found

    def components(self) -> list[list[str]]:
        """Connected components, largest first, ignoring edge direction."""
        undirected = self._graph.to_undirected(as_view=True)
        return sorted(
            (sorted(c) for c in nx.connected_components(undirected)),
            key=len,
            reverse=True,
        )

    def component_count(self) -> int:
        return nx.number_connected_components(self._graph.to_undirected(as_view=True))

    # --- serialisation for the graph visualisation ---------------------------

    def to_dict(self) -> dict[str, list[dict[str, object]]]:
        """JSON-ready nodes and edges for the canvas renderer.

        Node type travels with the node because the frontend colours by entity
        type, and re-deriving that from the name in JavaScript would put a
        second copy of the ontology in the browser.
        """
        return {
            "nodes": [
                {"id": name, "type": self.type_of(name)} for name in sorted(self._graph)
            ],
            "edges": [
                {
                    "source": e.subject,
                    "target": e.object,
                    "predicate": e.predicate,
                    "label": self._label(e.predicate, forward=True),
                    "sentence": e.sentence,
                    "doc_id": e.doc_id,
                    "doc_ids": list(e.doc_ids),
                }
                for e in sorted(
                    self.edges, key=lambda e: (e.subject, e.predicate, e.object)
                )
            ],
        }

    # --- internals -----------------------------------------------------------

    def _steps(self, node: str) -> Iterator[tuple[str, Edge, bool]]:
        """Every edge incident to `node`, outbound and inbound alike.

        Sorted so that traversal order depends on the graph rather than on the
        order the triples happened to be extracted in.
        """
        steps = [
            (target, data["edge"], True)
            for _, target, data in self._graph.out_edges(node, data=True)
        ] + [
            (source, data["edge"], False)
            for source, _, data in self._graph.in_edges(node, data=True)
        ]
        return iter(sorted(steps, key=lambda s: (s[0], s[1].predicate)))

    def _hop(self, source: str, target: str, edge: Edge, forward: bool) -> Hop:
        return Hop(
            source=source,
            target=target,
            label=self._label(edge.predicate, forward=True),
            forward=forward,
            edge=edge,
        )

    def _label(self, predicate: str, forward: bool) -> str:
        relation = self.ontology.relations[predicate]
        return relation.label if forward else relation.inverse_label
