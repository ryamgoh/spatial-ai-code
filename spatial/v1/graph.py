"""Stable graph primitives for extensible spatial dataset generators."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection


ENTITY_NAMES = (
    "Police Station",
    "Library",
    "Coffee Shop",
    "Supermarket",
    "Hospital",
    "Post Office",
    "Fire Department",
    "Museum",
    "Park",
    "Gas Station",
    "High School",
    "Cinema",
    "University",
    "City Hall",
    "Zoo",
    "Shopping Mall",
    "Bakery",
    "Church",
    "Bank",
    "Pharmacy",
)


class AxisGraph:
    """One-axis directed-order graph used to build symbolic SFT traces."""

    def __init__(self) -> None:
        self.nodes: set[str] = set()
        self.edges: set[tuple[str, str]] = set()

    def add_relation(self, lower: str, higher: str) -> None:
        self.nodes.add(lower)
        self.nodes.add(higher)
        self.edges.add((lower, higher))

    def get_transitive_closure(self) -> set[tuple[str, str]]:
        closure = set(self.edges)
        added = True
        while added:
            added = False
            new_edges: set[tuple[str, str]] = set()
            for a, b in closure:
                for c, d in closure:
                    if b == c and (a, d) not in closure:
                        new_edges.add((a, d))
                        added = True
            closure.update(new_edges)
        return closure

    def format_state(self, active_nodes: Collection[str] | None = None) -> str:
        target_nodes = set(active_nodes) if active_nodes is not None else self.nodes
        if not target_nodes:
            return "Empty"

        global_closure = self.get_transitive_closure()
        sub_closure = {
            (a, b)
            for a, b in global_closure
            if a in target_nodes and b in target_nodes
        }
        sub_reduction = set(sub_closure)
        for a, b in list(sub_reduction):
            for c in target_nodes:
                if (a, c) in sub_closure and (c, b) in sub_closure:
                    sub_reduction.discard((a, b))

        if not sub_reduction:
            return ", ".join(sorted(target_nodes))

        predecessors: dict[str, set[str]] = defaultdict(set)
        successors: dict[str, set[str]] = defaultdict(set)
        for a, b in sub_closure:
            predecessors[b].add(a)
            successors[a].add(b)

        groups: dict[tuple[frozenset[str], frozenset[str]], list[str]] = defaultdict(list)
        for node in target_nodes:
            key = (frozenset(predecessors[node]), frozenset(successors[node]))
            groups[key].append(node)

        node_to_group: dict[str, str] = {}
        for members in groups.values():
            sorted_members = sorted(members)
            name = (
                sorted_members[0]
                if len(sorted_members) == 1
                else "{" + ", ".join(sorted_members) + "}"
            )
            for member in members:
                node_to_group[member] = name

        group_edges = {
            (node_to_group[a], node_to_group[b])
            for a, b in sub_reduction
            if node_to_group[a] != node_to_group[b]
        }
        if not group_edges:
            return ", ".join(sorted(set(node_to_group.values())))

        next_nodes: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = defaultdict(int)
        nodes_in_edges: set[str] = set()
        for source, target in group_edges:
            next_nodes[source].append(target)
            in_degree[target] += 1
            in_degree.setdefault(source, 0)
            nodes_in_edges.update((source, target))

        chains: list[str] = []

        def build_path(current: str, path: list[str]) -> None:
            neighbors = sorted(next_nodes[current])
            if not neighbors:
                chains.append(" < ".join(path))
                return
            for neighbor in neighbors:
                build_path(neighbor, path + [neighbor])

        for start in sorted(node for node, degree in in_degree.items() if degree == 0):
            build_path(start, [start])
        chains.extend(sorted(set(node_to_group.values()) - nodes_in_edges))
        return ", ".join(sorted(set(chains)))
