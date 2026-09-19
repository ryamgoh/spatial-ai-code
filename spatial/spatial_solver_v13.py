"""Standalone solver and structural analyser for synthetic v13 data.

The v13 semantic contract is implemented here independently of older dataset
versions. The generator must round-trip every rendered prompt through this
module; its internal scene is never the final source of gold.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

RELATION_RE = re.compile(
    r"([^.]+?) is to the "
    r"(Northeast|Northwest|Southeast|Southwest|North|South|East|West) "
    r"of ([^.]+?)\.",
    re.IGNORECASE,
)
_UNDETERMINED = {
    "cannot be determined",
    "undetermined",
    "not enough information",
}
_NONE_OF_OPTIONS = {
    "none of the options",
    "none of these options",
    "none of the given options",
    "not among the options",
    "not among the given options",
    "no listed option is correct",
    "none of the above",
    "none of these",
}

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


def _letter_set(answer: str) -> set[str]:
    return {
        token.strip()
        for token in re.split(r"[,;| ]+", answer.strip())
        if token.strip() in {"A", "B", "C", "D", "E", "F"}
    }


@dataclass
class Grade:
    """V13 grading verdict for one rendered prompt."""

    raw: str
    q_type: int = -1
    options: dict[str, str] = field(default_factory=dict)
    verdicts: list[tuple[str, str, bool, str]] = field(default_factory=list)
    x_rel: str = "unknown"
    y_rel: str = "unknown"
    x_conflict: bool = False
    y_conflict: bool = False
    error: str | None = None

    @property
    def letters(self) -> set[str]:
        if self.raw.startswith("Error") or self.raw == "No valid options found":
            return set()
        return _letter_set(self.raw)

    @property
    def pretty(self) -> str:
        return ", ".join(key for key, _, accepted, _ in self.verdicts if accepted)

    @property
    def accept(self) -> bool:
        if self.error or self.raw.startswith("Error"):
            return False
        if self.raw == "No valid options found" or not self.letters:
            return False
        if self.q_type == 0 and len(self.letters) >= 4:
            return False
        return True


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


class SpatialSolverV13:
    """Standalone v13 parser, grader, and structural analyser."""

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

    @staticmethod
    def detect_type(question_part: str) -> int:
        if "In which direction is" in question_part:
            return 0
        if "Which object is in the" in question_part:
            return 1
        if "How many objects are in the" in question_part:
            return 2
        return -1

    @staticmethod
    def parse_options(question_part: str) -> dict[str, str]:
        options: dict[str, str] = {}
        tail = question_part.split("Available options", 1)[-1]
        if re.search(r"\b[A-F]\.\s*[^,\n]+,\s*[A-F]\.", tail):
            for match in re.finditer(
                r"\b([A-F])\.\s*(.+?)(?=\s*,\s*[A-F]\.|\s*$)", tail
            ):
                options[match.group(1)] = match.group(2).strip().rstrip(".,")
            return options
        for match in re.finditer(r"\b([A-F])\.\s*([^\n]+)", question_part):
            options[match.group(1)] = match.group(2).strip().rstrip(".")
        return options

    @staticmethod
    def is_undetermined_option(value: str) -> bool:
        return value.strip().rstrip(".").lower() in _UNDETERMINED

    @staticmethod
    def is_none_of_above(value: str) -> bool:
        return value.strip().rstrip(".").lower() in _NONE_OF_OPTIONS

    @staticmethod
    def _transitive_closure(
        objects: list[str], edges: list[tuple[str, str]]
    ) -> dict[str, set[str]]:
        reach: dict[str, set[str]] = {obj: set() for obj in objects}
        for source, target in edges:
            reach.setdefault(source, set()).add(target)
            reach.setdefault(target, set())
        changed = True
        while changed:
            changed = False
            for source in list(reach):
                before = len(reach[source])
                additions: set[str] = set()
                for target in reach[source]:
                    additions |= reach.get(target, set())
                reach[source] |= additions
                changed |= len(reach[source]) > before
        return reach

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
    def axis_status(
        reach: dict[str, set[str]], target: str, reference: str
    ) -> tuple[str, bool]:
        greater = reference in reach.get(target, set())
        lesser = target in reach.get(reference, set())
        if greater and lesser:
            return "unknown", True
        if greater:
            return "gt", False
        if lesser:
            return "lt", False
        return "unknown", False

    @classmethod
    def get_rel(cls, reach: dict[str, set[str]], a: str, b: str) -> str:
        relation, _ = cls.axis_status(reach, a, b)
        return relation

    def _in_axis(
        self,
        x_reach: dict[str, set[str]],
        y_reach: dict[str, set[str]],
        candidate: str,
        reference: str,
        axis: str,
    ) -> bool:
        expected = {
            "north": (y_reach, "gt"),
            "south": (y_reach, "lt"),
            "east": (x_reach, "gt"),
            "west": (x_reach, "lt"),
        }.get(axis)
        return bool(
            expected
            and self.get_rel(expected[0], candidate, reference) == expected[1]
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
        """Apply the standalone v13 semantic contract to one prompt."""
        objects, relations, question_part = self._parse_indexed_relations(text)
        cycle_profile = self._global_cycle_profile(relations, question_part)
        options = self.parse_options(question_part)
        q_type = self.detect_type(question_part)
        if not question_part:
            return Grade(raw="Error: no question found", error="no question found")
        if not options:
            return Grade(
                raw="Error: no options found", error="no options found", q_type=q_type
            )
        if cycle_profile["world_consistency"] == "consistent":
            return self._grade_consistent(objects, relations, question_part, options, q_type)
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

    def _grade_consistent(
        self,
        objects: list[str],
        relations: list[tuple[str, str, str, int]],
        question_part: str,
        options: dict[str, str],
        q_type: int,
    ) -> Grade:
        x_reach, y_reach = self.build_order_graphs(objects, relations)
        object_set = set(objects)
        valid: list[str] = []
        verdicts: list[tuple[str, str, bool, str]] = []
        x_rel = y_rel = "unknown"

        if q_type == 0:
            match = re.search(
                r"In which direction is ([^?]+?) relative to ([^?]+?)\?",
                question_part,
            )
            if not match:
                return Grade(
                    raw="Error: cannot parse type-0 question",
                    error="parse type-0",
                    q_type=0,
                )
            target = _strip_the(match.group(1))
            reference = _strip_the(match.group(2))
            x_rel, _ = self.axis_status(x_reach, target, reference)
            y_rel, _ = self.axis_status(y_reach, target, reference)
            possible = self._possible_directions(x_rel, y_rel)
            listed = [
                key for key in sorted(options) if options[key] in possible
            ]
            listed_values = {options[key] for key in listed}
            incomplete = (
                len(possible) == 2
                and bool(listed_values)
                and listed_values != set(possible)
            )
            if not possible or incomplete:
                valid = [
                    key
                    for key in sorted(options)
                    if self.is_undetermined_option(options[key])
                ]
            elif listed_values == set(possible):
                valid = listed
            else:
                valid = [
                    key
                    for key in sorted(options)
                    if self.is_none_of_above(options[key])
                ]
            for key in sorted(options):
                value = options[key]
                accepted = key in valid
                if accepted and self.is_undetermined_option(value):
                    reason = (
                        "only one of the two possible compounds is listed"
                        if incomplete
                        else "neither axis is logically determined"
                    )
                elif accepted and self.is_none_of_above(value):
                    reason = "the proven direction is not listed"
                elif accepted:
                    reason = "matches every derived axis component"
                elif value in possible:
                    reason = "not selectable under the complete option-set rule"
                else:
                    reason = "does not match the derived axis information"
                verdicts.append((key, value, accepted, reason))

        elif q_type in (1, 2):
            pattern = (
                r"Which object is in the (\w+) of ([^?]+?)\?"
                if q_type == 1
                else r"How many objects are in the (\w+) of ([^?]+?)\?"
            )
            match = re.search(pattern, question_part)
            if not match:
                return Grade(
                    raw=f"Error: cannot parse type-{q_type} question",
                    error=f"parse type-{q_type}",
                    q_type=q_type,
                )
            direction = match.group(1).strip().title()
            reference = _strip_the(match.group(2))
            requirements = _DIRECTION_REQUIREMENTS.get(direction)
            if not requirements:
                return Grade(
                    raw="Error: unknown direction",
                    error="unknown direction",
                    q_type=q_type,
                )
            proven = sorted(
                obj
                for obj in objects
                if obj != reference
                and all(
                    self.get_rel(
                        x_reach if axis == "x" else y_reach, obj, reference
                    )
                    == required_relation
                    for axis, required_relation in requirements
                )
            )
            if q_type == 1:
                for key in sorted(options):
                    value = options[key]
                    candidate = _strip_the(value)
                    if self.is_undetermined_option(value) or self.is_none_of_above(value):
                        continue
                    accepted = candidate in object_set and candidate in proven
                    verdicts.append(
                        (
                            key,
                            value,
                            accepted,
                            (
                                f"proven on every required axis of {direction}"
                                if accepted
                                else f"not proven on every required axis of {direction}"
                            ),
                        )
                    )
                    if accepted:
                        valid.append(key)
            else:
                count = len(proven)
                for key in sorted(options):
                    value = options[key]
                    if self.is_undetermined_option(value) or self.is_none_of_above(value):
                        continue
                    try:
                        accepted = int(value.strip()) == count
                    except (TypeError, ValueError):
                        accepted = False
                    verdicts.append(
                        (
                            key,
                            value,
                            accepted,
                            f"matches the derived count {count}"
                            if accepted
                            else f"count is {count}, not {value}",
                        )
                    )
                    if accepted:
                        valid.append(key)
            if not valid:
                valid = [
                    key
                    for key in sorted(options)
                    if self.is_none_of_above(options[key])
                ]
            seen = {key for key, *_ in verdicts}
            for key in sorted(options):
                if key in seen:
                    continue
                value = options[key]
                accepted = key in valid
                if self.is_none_of_above(value):
                    reason = (
                        "no listed answer is logically proven"
                        if accepted
                        else "a listed answer is logically proven"
                    )
                elif self.is_undetermined_option(value):
                    reason = "the world is consistent; this option is reserved for inconsistency"
                else:
                    reason = "not a valid content option"
                verdicts.append((key, value, accepted, reason))
        else:
            return Grade(
                raw="Error: unknown question type",
                error="unknown type",
                q_type=q_type,
            )

        raw = ",".join(valid) if valid else "No valid options found"
        return Grade(
            raw=raw,
            q_type=q_type,
            options=options,
            verdicts=verdicts,
            x_rel=x_rel,
            y_rel=y_rel,
        )

    @staticmethod
    def _possible_directions(x_rel: str, y_rel: str) -> list[str]:
        x_values = ("East",) if x_rel == "gt" else ("West",) if x_rel == "lt" else ("East", "West")
        y_values = ("North",) if y_rel == "gt" else ("South",) if y_rel == "lt" else ("North", "South")
        if x_rel == y_rel == "unknown":
            return []
        return [f"{y}{x.lower()}" for y in y_values for x in x_values]

    def solve(self, text: str) -> str:
        return self.grade(text).raw

    def agrees(self, text: str, answer: str) -> bool:
        grade = self.grade(text)
        return grade.accept and grade.letters == _letter_set(answer)

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
        grade = self.grade(text)
        cycle_profile = self._global_cycle_profile(relations, question_part)
        question_family = {0: "direction", 1: "which", 2: "count"}.get(
            grade.q_type, "unknown"
        )
        # An inconsistent world has no meaningful local query semantics. The
        # generator may attach its pre-injection, solver-verified base subtype
        # to a dataset row, but the prompt-only analyser must not invent one.
        semantic_subtype = (
            None
            if cycle_profile["world_consistency"] == "inconsistent"
            else self._semantic_subtype(grade)
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
            "question_family": question_family,
            "semantic_subtype": semantic_subtype,
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
