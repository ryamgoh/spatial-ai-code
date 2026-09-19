"""Versioned solver and structural analyser for synthetic v13 data.

v13 preserves the v6 multiple-choice laws while extending the relation
language with the four cardinal directions.  The generator must round-trip
every rendered prompt through this module; its internal scene is never the
final source of gold.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any

from spatial_solver import Grade, SpatialSolver, _normalize_text, _strip_the


RELATION_RE = re.compile(
    r"([^.]+?) is to the "
    r"(Northeast|Northwest|Southeast|Southwest|North|South|East|West) "
    r"of ([^.]+?)\.",
    re.IGNORECASE,
)


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
                "path": reverse_path,
                "supporting_statements": reverse_support,
            }
        return {
            "relation": "unknown",
            "conflict": False,
            "depth": None,
            "path": None,
            "supporting_statements": [],
        }

    def analyze(self, text: str) -> dict[str, Any]:
        """Describe the actual structure recovered from a rendered prompt."""
        objects, relations, question_part = self._parse_indexed_relations(text)
        grade = self.grade(text)
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
            "semantic_subtype": self._semantic_subtype(grade),
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
            "x_conflict": False,
            "y_conflict": False,
        }

        query = re.search(
            r"In which direction is ([^?]+?) relative to ([^?]+?)\?",
            question_part,
        )
        if not query:
            return result
        target = _strip_the(query.group(1))
        reference = _strip_the(query.group(2))
        x_edges: list[tuple[str, str, int]] = []
        y_edges: list[tuple[str, str, int]] = []
        for a, direction, b, statement_index in relations:
            x_part, y_part = _axis_edges(a, direction, b)
            x_edges.extend((source, dest, statement_index) for source, dest in x_part)
            y_edges.extend((source, dest, statement_index) for source, dest in y_part)

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
        return result

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
