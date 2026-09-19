"""Versioned solver and structural analyser for synthetic v13 data.

v13 preserves the v6 multiple-choice laws while extending the relation
language with the four cardinal directions.  The generator must round-trip
every rendered prompt through this module; its internal scene is never the
final source of gold.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Any

from spatial_solver import Grade, SpatialSolver


RELATION_RE = re.compile(
    r"([^.]+?) is to the "
    r"(Northeast|Northwest|Southeast|Southwest|North|South|East|West) "
    r"of ([^.]+?)\.",
    re.IGNORECASE,
)

_DIRECTION_REQUIREMENTS = {
    "North": (("y", "gt"),),
    "South": (("y", "lt"),),
    "East": (("x", "gt"),),
    "West": (("x", "lt"),),
    "Northeast": (("x", "gt"), ("y", "gt")),
    "Northwest": (("x", "lt"), ("y", "gt")),
    "Southeast": (("x", "gt"), ("y", "lt")),
    "Southwest": (("x", "lt"), ("y", "lt")),
}


def _normalize_text(text: str) -> str:
    return (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def _strip_the(name: str) -> str:
    normalized = name.strip().strip(".")
    if normalized.lower().startswith("the "):
        return normalized[4:].strip()
    return normalized


@dataclass(frozen=True)
class SolvedProblem:
    """One authoritative parse/grade/structure result for a rendered prompt."""

    grade: Grade
    structure: dict[str, Any]
    objects: tuple[str, ...]
    relations: tuple[tuple[str, str, str], ...]
    question_part: str


def _axis_edges(a: str, direction: str, b: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return solver-oriented edges: greater/east/north node -> lesser node."""
    x_edges: list[tuple[str, str]] = []
    y_edges: list[tuple[str, str]] = []
    if direction in {"northeast", "southeast", "east"}:
        x_edges.append((a, b))
    elif direction in {"northwest", "southwest", "west"}:
        x_edges.append((b, a))
    if direction in {"northeast", "northwest", "north"}:
        y_edges.append((a, b))
    elif direction in {"southeast", "southwest", "south"}:
        y_edges.append((b, a))
    return x_edges, y_edges


class SpatialSolverV13(SpatialSolver):
    """v6-compatible grader with cardinal parsing and proof metadata."""

    def _parse_indexed_relations(
        self, text: str
    ) -> tuple[list[str], list[tuple[str, str, str, int]], str]:
        text = _normalize_text(text)
        map_part, question_part = text, ""
        for marker in ("Please answer", "Question:"):
            if marker in text:
                index = text.index(marker)
                map_part = text[:index]
                question_part = text[index:]
                break
        map_part = re.sub(
            r"^Consider a map with multiple (?:locations|objects):\s*",
            "",
            map_part,
        ).lstrip()

        objects: set[str] = set()
        standalone = re.search(r"([^\n.]+?) is in the map", map_part)
        if standalone:
            objects.add(_strip_the(standalone.group(1)))

        relations: list[tuple[str, str, str, int]] = []
        for statement_index, match in enumerate(RELATION_RE.finditer(map_part)):
            a = _strip_the(match.group(1))
            direction = match.group(2).strip().lower()
            b = _strip_the(match.group(3))
            relations.append((a, direction, b, statement_index))
            objects.update((a, b))
        return sorted(objects), relations, question_part

    def parse_problem(self, text: str):
        objects, indexed, question_part = self._parse_indexed_relations(text)
        return objects, [(a, direction, b) for a, direction, b, _ in indexed], question_part

    def build_order_graphs(self, objects: list[str], relations: list[tuple]):
        x_edges: list[tuple[str, str]] = []
        y_edges: list[tuple[str, str]] = []
        for relation in relations:
            a, direction, b = relation[:3]
            x_part, y_part = _axis_edges(a, direction, b)
            x_edges.extend(x_part)
            y_edges.extend(y_part)
        return (
            self._transitive_closure(objects, x_edges),
            self._transitive_closure(objects, y_edges),
        )

    @staticmethod
    def _find_cycle(
        edges: list[tuple[str, str, int]],
    ) -> tuple[list[str], list[int]] | None:
        """Return one closed directed-cycle witness and its statement ids."""
        adjacency: dict[str, list[tuple[str, int]]] = {}
        for source, target, statement_index in edges:
            adjacency.setdefault(source, []).append((target, statement_index))
        visited: set[str] = set()
        active: set[str] = set()
        node_stack: list[str] = []
        edge_stack: list[int] = []

        def visit(node: str) -> tuple[list[str], list[int]] | None:
            visited.add(node)
            active.add(node)
            node_stack.append(node)
            for neighbor, statement_index in sorted(adjacency.get(node, [])):
                if neighbor not in visited:
                    edge_stack.append(statement_index)
                    found = visit(neighbor)
                    if found:
                        return found
                    edge_stack.pop()
                elif neighbor in active:
                    cycle_start = node_stack.index(neighbor)
                    return (
                        node_stack[cycle_start:] + [neighbor],
                        edge_stack[cycle_start:] + [statement_index],
                    )
            node_stack.pop()
            active.remove(node)
            return None

        for node in sorted(adjacency):
            if node not in visited:
                found = visit(node)
                if found:
                    return found
        return None

    @staticmethod
    def _query_anchors(question_part: str) -> set[str]:
        direction_query = re.search(
            r"In which direction is ([^?]+?) relative to ([^?]+?)\?",
            question_part,
        )
        if direction_query:
            return {
                _strip_the(direction_query.group(1)),
                _strip_the(direction_query.group(2)),
            }
        set_query = re.search(
            r"(?:Which object is|How many objects are) in the \w+ of ([^?]+?)\?",
            question_part,
        )
        return {_strip_the(set_query.group(1))} if set_query else set()

    @staticmethod
    def _connected_nodes(
        relations: list[tuple[str, str, str, int]], anchors: set[str]
    ) -> set[str]:
        adjacency: dict[str, set[str]] = {}
        for a, _direction, b, _statement_index in relations:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        connected = set(anchors)
        frontier = list(anchors)
        while frontier:
            node = frontier.pop()
            for neighbor in adjacency.get(node, set()):
                if neighbor not in connected:
                    connected.add(neighbor)
                    frontier.append(neighbor)
        return connected

    def _global_cycle_profile(
        self,
        relations: list[tuple[str, str, str, int]],
        question_part: str,
    ) -> dict[str, Any]:
        x_edges: list[tuple[str, str, int]] = []
        y_edges: list[tuple[str, str, int]] = []
        for a, direction, b, statement_index in relations:
            x_part, y_part = _axis_edges(a, direction, b)
            x_edges.extend(
                (source, target, statement_index) for source, target in x_part
            )
            y_edges.extend(
                (source, target, statement_index) for source, target in y_part
            )
        witnesses = {
            axis: witness
            for axis, edges in (("x", x_edges), ("y", y_edges))
            if (witness := self._find_cycle(edges)) is not None
        }
        axes = sorted(witnesses)
        statement_indices = sorted(
            {index for _nodes, indices in witnesses.values() for index in indices}
        )
        cycle_nodes = {node for nodes, _indices in witnesses.values() for node in nodes}
        connected = self._connected_nodes(relations, self._query_anchors(question_part))
        placement = (
            "none"
            if not axes
            else "query-connected"
            if cycle_nodes & connected
            else "disconnected"
        )
        cycle_lengths = [len(indices) for _nodes, indices in witnesses.values()]
        topology = (
            "none"
            if not cycle_lengths
            else "direct"
            if max(cycle_lengths) == 2
            else "indirect"
        )
        return {
            "world_consistency": "inconsistent" if axes else "consistent",
            "cycle_axes": axes,
            "cycle_topology": topology,
            "cycle_placement": placement,
            "cycle_statement_indices": statement_indices,
            "cycle_lengths": {
                axis: len(indices)
                for axis, (_nodes, indices) in witnesses.items()
            },
            "cycle_witnesses": {
                axis: nodes for axis, (nodes, _indices) in witnesses.items()
            },
        }

    def grade(self, text: str) -> Grade:
        """Apply global world consistency before the inherited query rules."""
        objects, relations, question_part = self._parse_indexed_relations(text)
        cycle_profile = self._global_cycle_profile(relations, question_part)
        if cycle_profile["world_consistency"] == "consistent":
            return super().grade(text)
        options = self.parse_options(question_part)
        q_type = self.detect_type(question_part)
        gold_keys = [
            key
            for key in sorted(options)
            if self.is_undetermined_option(options[key])
        ]
        verdicts = [
            (
                key,
                value,
                key in gold_keys,
                (
                    "the complete premise set is inconsistent"
                    if key in gold_keys
                    else "the world is inconsistent, so no content answer is reliable"
                ),
            )
            for key, value in sorted(options.items())
        ]
        raw = ",".join(gold_keys) if gold_keys else "No valid options found"
        return Grade(
            raw=raw,
            q_type=q_type,
            options=options,
            verdicts=verdicts,
            x_rel="unknown",
            y_rel="unknown",
            x_conflict="x" in cycle_profile["cycle_axes"],
            y_conflict="y" in cycle_profile["cycle_axes"],
        )

    @staticmethod
    def _shortest_path(
        edges: list[tuple[str, str, int]], start: str, end: str
    ) -> tuple[list[str] | None, list[int]]:
        adjacency: dict[str, list[tuple[str, int]]] = {}
        for source, target, statement_index in edges:
            adjacency.setdefault(source, []).append((target, statement_index))
        queue = deque([(start, [start], [])])
        visited = {start}
        while queue:
            node, path, supporting = queue.popleft()
            if node == end:
                return path, supporting
            for nxt, statement_index in sorted(adjacency.get(node, [])):
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, path + [nxt], supporting + [statement_index]))
        return None, []

    def _axis_proof(
        self,
        edges: list[tuple[str, str, int]],
        target: str,
        reference: str,
    ) -> dict[str, Any]:
        forward_path, forward_support = self._shortest_path(edges, target, reference)
        reverse_path, reverse_support = self._shortest_path(edges, reference, target)
        if forward_path and reverse_path:
            return {
                "relation": "unknown",
                "conflict": True,
                "depth": None,
                "path": None,
                "supporting_statements": sorted(set(forward_support + reverse_support)),
            }
        if forward_path:
            # Present paths in increasing axis order (reference -> target).
            return {
                "relation": "gt",
                "conflict": False,
                "depth": len(forward_path) - 1,
                "path": list(reversed(forward_path)),
                "supporting_statements": forward_support,
            }
        if reverse_path:
            return {
                "relation": "lt",
                "conflict": False,
                "depth": len(reverse_path) - 1,
                # Edges point greater -> lesser; render every proof in the
                # state notation's lower -> higher order.
                "path": list(reversed(reverse_path)),
                "supporting_statements": reverse_support,
            }
        return {
            "relation": "unknown",
            "conflict": False,
            "depth": None,
            "path": None,
            "supporting_statements": [],
        }

    @staticmethod
    def _distractor_profile(
        relations: list[tuple[str, str, str, int]],
        relevant_statements: set[int],
        proof_nodes: set[str],
    ) -> dict[str, Any]:
        """Classify non-proof relations by connectivity to the query proof."""
        adjacency: dict[str, set[str]] = {}
        for a, _direction, b, _statement_index in relations:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

        query_component = set(proof_nodes)
        frontier = list(proof_nodes)
        while frontier:
            node = frontier.pop()
            for neighbor in adjacency.get(node, set()):
                if neighbor not in query_component:
                    query_component.add(neighbor)
                    frontier.append(neighbor)

        distractors = [
            statement_index
            for _a, _direction, _b, statement_index in relations
            if statement_index not in relevant_statements
        ]
        query_branch = [
            statement_index
            for a, _direction, b, statement_index in relations
            if statement_index in distractors
            and (a in query_component or b in query_component)
        ]
        disconnected = sorted(set(distractors) - set(query_branch))
        return {
            "relevant_statement_indices": sorted(relevant_statements),
            "distractor_statement_indices": sorted(distractors),
            "num_relevant_relations": len(relevant_statements),
            "num_distractor_relations": len(distractors),
            "disconnected_distractor_statement_indices": disconnected,
            "query_branch_distractor_statement_indices": sorted(query_branch),
            "num_disconnected_distractors": len(disconnected),
            "num_query_branch_distractors": len(query_branch),
        }

    def solve_and_analyze(self, text: str) -> SolvedProblem:
        """Grade and structurally analyse a rendered prompt in one pass."""
        objects, relations, question_part = self._parse_indexed_relations(text)
        local_grade = super().grade(text)
        grade = self.grade(text)
        cycle_profile = self._global_cycle_profile(relations, question_part)
        local_subtype = self._semantic_subtype(local_grade)
        final_subtype = (
            {0: "dir-cycle", 1: "which-cycle", 2: "count-cycle"}.get(
                grade.q_type, "unknown-cycle"
            )
            if cycle_profile["world_consistency"] == "inconsistent"
            else local_subtype
        )
        directions = [direction for _, direction, _, _ in relations]
        cardinal_count = sum(direction in {"north", "south", "east", "west"} for direction in directions)
        diagonal_count = len(directions) - cardinal_count
        if cardinal_count and diagonal_count:
            relation_mix = "mixed"
        elif cardinal_count:
            relation_mix = "cardinal-only"
        elif diagonal_count:
            relation_mix = "diagonal-only"
        else:
            relation_mix = "none"

        result: dict[str, Any] = {
            "question_type": self.detect_type(question_part),
            "local_semantic_subtype": local_subtype,
            "semantic_subtype": final_subtype,
            "num_entities": len(objects),
            "num_relations": len(relations),
            "num_cardinal_relations": cardinal_count,
            "num_diagonal_relations": diagonal_count,
            "relation_mix": relation_mix,
            "x_depth": None,
            "y_depth": None,
            "x_direct": False,
            "y_direct": False,
            "x_path": None,
            "y_path": None,
            "x_supporting_statements": [],
            "y_supporting_statements": [],
            "shared_supporting_statements": [],
            "axes_independent": None,
            "x_conflict": "x" in cycle_profile["cycle_axes"],
            "y_conflict": "y" in cycle_profile["cycle_axes"],
            "relevant_statement_indices": [],
            "distractor_statement_indices": [],
            "num_relevant_relations": 0,
            "num_distractor_relations": 0,
            "disconnected_distractor_statement_indices": [],
            "query_branch_distractor_statement_indices": [],
            "num_disconnected_distractors": 0,
            "num_query_branch_distractors": 0,
            **cycle_profile,
        }
        x_edges: list[tuple[str, str, int]] = []
        y_edges: list[tuple[str, str, int]] = []
        for a, direction, b, statement_index in relations:
            x_part, y_part = _axis_edges(a, direction, b)
            x_edges.extend(
                (source, dest, statement_index) for source, dest in x_part
            )
            y_edges.extend(
                (source, dest, statement_index) for source, dest in y_part
            )

        if cycle_profile["world_consistency"] == "inconsistent":
            return SolvedProblem(
                grade=grade,
                structure=result,
                objects=tuple(objects),
                relations=tuple(
                    (a, direction, b) for a, direction, b, _ in relations
                ),
                question_part=question_part,
            )

        query = re.search(
            r"In which direction is ([^?]+?) relative to ([^?]+?)\?",
            question_part,
        )
        if not query:
            set_query = re.search(
                r"(?:Which object is|How many objects are) in the (\w+) of ([^?]+?)\?",
                question_part,
            )
            if set_query:
                direction = set_query.group(1).strip().title()
                reference = _strip_the(set_query.group(2))
                requirements = _DIRECTION_REQUIREMENTS.get(direction, ())
                proven_entity_proofs = []
                for entity in sorted(obj for obj in objects if obj != reference):
                    proofs = {
                        "x": self._axis_proof(x_edges, entity, reference),
                        "y": self._axis_proof(y_edges, entity, reference),
                    }
                    if all(
                        proofs[axis]["relation"] == required_relation
                        for axis, required_relation in requirements
                    ):
                        proof = {"entity": entity}
                        for axis, _ in requirements:
                            proof[f"{axis}_path"] = proofs[axis]["path"]
                            proof[f"{axis}_depth"] = proofs[axis]["depth"]
                            proof[f"{axis}_supporting_statements"] = proofs[axis]["supporting_statements"]
                        proven_entity_proofs.append(proof)
                result.update(
                    {
                        "query_reference": reference,
                        "query_direction": direction,
                        "required_axes": [axis for axis, _ in requirements],
                        "proven_entities": [
                            proof["entity"] for proof in proven_entity_proofs
                        ],
                        "proven_entity_proofs": proven_entity_proofs,
                    }
                )
                relevant = {
                    statement_index
                    for proof in proven_entity_proofs
                    for axis, _ in requirements
                    for statement_index in proof[f"{axis}_supporting_statements"]
                }
                proof_nodes = {reference} | {
                    node
                    for proof in proven_entity_proofs
                    for axis, _ in requirements
                    for node in proof[f"{axis}_path"]
                }
                result.update(
                    self._distractor_profile(relations, relevant, proof_nodes)
                )
            return SolvedProblem(
                grade=grade,
                structure=result,
                objects=tuple(objects),
                relations=tuple(
                    (a, direction, b) for a, direction, b, _ in relations
                ),
                question_part=question_part,
            )
        target = _strip_the(query.group(1))
        reference = _strip_the(query.group(2))
        x_proof = self._axis_proof(x_edges, target, reference)
        y_proof = self._axis_proof(y_edges, target, reference)
        shared = sorted(
            set(x_proof["supporting_statements"])
            & set(y_proof["supporting_statements"])
        )
        both_proven = x_proof["depth"] is not None and y_proof["depth"] is not None
        result.update(
            {
                "target": target,
                "reference": reference,
                "x_depth": x_proof["depth"],
                "y_depth": y_proof["depth"],
                "x_direct": x_proof["depth"] == 1,
                "y_direct": y_proof["depth"] == 1,
                "x_path": x_proof["path"],
                "y_path": y_proof["path"],
                "x_supporting_statements": x_proof["supporting_statements"],
                "y_supporting_statements": y_proof["supporting_statements"],
                "shared_supporting_statements": shared,
                "axes_independent": (not shared) if both_proven else None,
                "x_conflict": x_proof["conflict"],
                "y_conflict": y_proof["conflict"],
            }
        )
        relevant = set(x_proof["supporting_statements"]) | set(
            y_proof["supporting_statements"]
        )
        proof_nodes = {target, reference}
        proof_nodes.update(x_proof["path"] or [])
        proof_nodes.update(y_proof["path"] or [])
        result.update(self._distractor_profile(relations, relevant, proof_nodes))
        return SolvedProblem(
            grade=grade,
            structure=result,
            objects=tuple(objects),
            relations=tuple(
                (a, direction, b) for a, direction, b, _ in relations
            ),
            question_part=question_part,
        )

    def analyze(self, text: str) -> dict[str, Any]:
        """Compatibility convenience returning only structural metadata."""
        return self.solve_and_analyze(text).structure

    def _semantic_subtype(self, grade: Grade) -> str:
        """Classify answer semantics from the solver verdict, not generator intent."""
        accepted = [
            value for _, value, is_accepted, _ in grade.verdicts if is_accepted
        ]
        if grade.q_type == 0:
            if grade.x_conflict or grade.y_conflict:
                return "dir-cycle"
            if any(self.is_none_of_above(value) for value in accepted):
                return "dir-omit"
            if any(self.is_undetermined_option(value) for value in accepted):
                one_axis_known = (grade.x_rel == "unknown") != (grade.y_rel == "unknown")
                return "dir-incomplete" if one_axis_known else "dir-undetermined"
            if len(grade.letters) == 1:
                return "dir-1"
            if len(grade.letters) == 2:
                return "dir-2"
            return "dir-unknown"
        if grade.q_type == 1:
            if any(
                self.is_none_of_above(value) or self.is_undetermined_option(value)
                for value in accepted
            ):
                return "which-0"
            return f"which-{len(grade.letters)}"
        if grade.q_type == 2:
            if any(self.is_none_of_above(value) for value in accepted):
                return "count-omit"
            return "count-1"
        return "unknown"


_SOLVER = SpatialSolverV13()


def solve(text: str) -> str:
    return _SOLVER.solve(text)


def grade(text: str) -> Grade:
    return _SOLVER.grade(text)


def analyze(text: str) -> dict[str, Any]:
    return _SOLVER.analyze(text)


def solve_and_analyze(text: str) -> SolvedProblem:
    return _SOLVER.solve_and_analyze(text)
