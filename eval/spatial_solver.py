"""Standalone v6 spatial solver.

Shared gold rules for:
  eval/clean_v6.py            — re-label SpatialMap JSONL
  finetune/generate_all_v6.py — proposes a map; this class is the only gold

Type 0: unique compound / two remaining compounds / E. Never A,B,C,D.
Type 1: only entities proven on every required axis. No all-four fallback.
Type 2: definite count.
A cycle on an axis makes that axis unknown for the pair (not a unique order).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_SPLIT_MARKERS = ("Please answer", "Question:")

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

_DIR_AXES_MAP = {
    "Northeast": ("north", "east"),
    "Northwest": ("north", "west"),
    "Southeast": ("south", "east"),
    "Southwest": ("south", "west"),
    "North": ("north",),
    "South": ("south",),
    "East": ("east",),
    "West": ("west",),
}


def _normalize_text(text: str) -> str:
    return (
        text
        .replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
    )


def _strip_the(name: str) -> str:
    name = name.strip().strip(".")
    if name.lower().startswith("the "):
        name = name[4:].strip()
    return name


def _letter_set(answer: str) -> set[str]:
    return {
        tok.strip()
        for tok in re.split(r"[,;| ]+", answer.strip())
        if tok.strip() in {"A", "B", "C", "D", "E", "F"}
    }


@dataclass
class Grade:
    """Solver verdict for one prompt. Generator gold must equal ``pretty``."""

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
        return ", ".join(k for k, _, inn, _ in self.verdicts if inn)

    @property
    def accept(self) -> bool:
        """Legal v6 gold: nonempty, not an error.

        Type 0 never marks all four compounds. Type 1 may mark 1–4 listed
        entities (N of the menu). Type 2 is one count or none-of-options.
        """
        if self.error or self.raw.startswith("Error"):
            return False
        if self.raw == "No valid options found" or not self.letters:
            return False
        if self.q_type == 0 and len(self.letters) >= 4:
            return False
        return True


class SpatialSolver:
    """Parse a SpatialMap / generator prompt and return v6 gold letters."""

    def parse_problem(self, text: str):
        """Return (objects, relations, question_part).

        relations: list of (a, dir, b) with dir lowercase
        (northeast/northwest/southeast/southwest).
        """
        text = _normalize_text(text)
        map_part, question_part = text, ""
        for marker in _SPLIT_MARKERS:
            if marker in text:
                idx = text.index(marker)
                map_part = text[:idx]
                question_part = text[idx:]
                break

        map_part = re.sub(
            r"^Consider a map with multiple (?:locations|objects):\s*",
            "",
            map_part,
        ).lstrip()

        objects: set[str] = set()
        m = re.search(r"([^\n.]+?) is in the map", map_part)
        if m:
            objects.add(_strip_the(m.group(1)))

        rel_re = re.compile(
            r"([^.]+?) is to the (Northeast|Northwest|Southeast|Southwest) of ([^.]+?)\.",
            re.IGNORECASE,
        )
        relations = []
        for m in rel_re.finditer(map_part):
            a = _strip_the(m.group(1))
            d = m.group(2).strip().lower()
            b = _strip_the(m.group(3))
            relations.append((a, d, b))
            objects.add(a)
            objects.add(b)

        return list(objects), relations, question_part

    def detect_type(self, question_part: str) -> int:
        if "In which direction is" in question_part:
            return 0
        if "Which object is in the" in question_part:
            return 1
        if "How many objects are in the" in question_part:
            return 2
        return -1

    def parse_options(self, question_part: str) -> dict[str, str]:
        opts = {}
        tail = question_part.split("Available options", 1)[-1]
        if re.search(r"\b[A-F]\.\s*[^,\n]+,\s*[A-F]\.", tail):
            for m in re.finditer(
                r"\b([A-F])\.\s*(.+?)(?=\s*,\s*[A-F]\.|\s*$)",
                tail,
            ):
                opts[m.group(1)] = m.group(2).strip().rstrip(".,")
            return opts
        for m in re.finditer(r"\b([A-F])\.\s*([^\n]+)", question_part):
            opts[m.group(1)] = m.group(2).strip().rstrip(".")
        return opts

    @staticmethod
    def is_undetermined_option(val: str) -> bool:
        return val.strip().rstrip(".").lower() in _UNDETERMINED

    @staticmethod
    def is_none_of_above(val: str) -> bool:
        return val.strip().rstrip(".").lower() in _NONE_OF_OPTIONS

    def _transitive_closure(self, objects: list[str], edges: list[tuple]) -> dict[str, set]:
        reach: dict[str, set] = {obj: set() for obj in objects}
        for a, b in edges:
            if a in reach:
                reach[a].add(b)
        changed = True
        while changed:
            changed = False
            for a in objects:
                before = len(reach[a])
                extras: set = set()
                for b in reach[a]:
                    extras |= reach.get(b, set())
                reach[a] |= extras
                if len(reach[a]) > before:
                    changed = True
        return reach

    def build_order_graphs(self, objects: list[str], relations: list[tuple]):
        """x_reach[a] contains b → a is east of b. y_reach[a] contains b → a is north of b."""
        x_edges, y_edges = [], []
        for a, d, b in relations:
            if d == "northeast":
                x_edges.append((a, b))
                y_edges.append((a, b))
            elif d == "southeast":
                x_edges.append((a, b))
                y_edges.append((b, a))
            elif d == "southwest":
                x_edges.append((b, a))
                y_edges.append((b, a))
            elif d == "northwest":
                x_edges.append((b, a))
                y_edges.append((a, b))
        return (
            self._transitive_closure(objects, x_edges),
            self._transitive_closure(objects, y_edges),
        )

    @staticmethod
    def axis_status(reach: dict, a: str, b: str) -> tuple[str, bool]:
        """Return (rel, conflict). rel is gt/lt/unknown. conflict if both ways."""
        gt = b in reach.get(a, set())
        lt = a in reach.get(b, set())
        if gt and lt:
            return "unknown", True
        if gt:
            return "gt", False
        if lt:
            return "lt", False
        return "unknown", False

    @staticmethod
    def get_rel(reach: dict, a: str, b: str) -> str:
        rel, _ = SpatialSolver.axis_status(reach, a, b)
        return rel

    def _in_axis(self, x_reach, y_reach, candidate: str, ref: str, axis: str) -> bool:
        if axis == "north":
            return self.get_rel(y_reach, candidate, ref) == "gt"
        if axis == "south":
            return self.get_rel(y_reach, candidate, ref) == "lt"
        if axis == "east":
            return self.get_rel(x_reach, candidate, ref) == "gt"
        if axis == "west":
            return self.get_rel(x_reach, candidate, ref) == "lt"
        return False

    def solve(self, text: str) -> str:
        """Return comma-separated gold letters, or an Error / empty-oracle string."""
        return self.grade(text).raw

    def grade(self, text: str) -> Grade:
        """Full verdict used as generator gold."""
        objects, relations, question_part = self.parse_problem(text)
        if not question_part:
            return Grade(raw="Error: no question found", error="no question found")

        q_type = self.detect_type(question_part)
        options = self.parse_options(question_part)
        if not options:
            return Grade(raw="Error: no options found", error="no options found", q_type=q_type)

        x_reach, y_reach = self.build_order_graphs(objects, relations)
        objects_set = set(objects)
        valid: list[str] = []
        verdicts: list[tuple[str, str, bool, str]] = []
        x_rel = y_rel = "unknown"
        x_conflict = y_conflict = False

        if q_type == 0:
            m = re.search(
                r"In which direction is ([^?]+?) relative to ([^?]+?)\?",
                question_part,
            )
            if not m:
                return Grade(raw="Error: cannot parse type-0 question", error="parse type-0", q_type=0)
            obj_x = _strip_the(m.group(1))
            obj_y = _strip_the(m.group(2))
            x_rel, x_conflict = self.axis_status(x_reach, obj_x, obj_y)
            y_rel, y_conflict = self.axis_status(y_reach, obj_x, obj_y)
            in_north = y_rel == "gt"
            in_south = y_rel == "lt"
            in_east = x_rel == "gt"
            in_west = x_rel == "lt"
            if in_north and in_east:
                possible_dirs = ["Northeast"]
            elif in_north and in_west:
                possible_dirs = ["Northwest"]
            elif in_south and in_east:
                possible_dirs = ["Southeast"]
            elif in_south and in_west:
                possible_dirs = ["Southwest"]
            elif in_north:
                possible_dirs = ["Northeast", "Northwest"]
            elif in_south:
                possible_dirs = ["Southeast", "Southwest"]
            elif in_east:
                possible_dirs = ["Northeast", "Southeast"]
            elif in_west:
                possible_dirs = ["Northwest", "Southwest"]
            else:
                possible_dirs = []
            both_unknown = not possible_dirs
            listed_live = [
                key for key in sorted(options)
                if options[key] in possible_dirs
            ]
            listed_live_dirs = {options[k] for k in listed_live}
            incomplete_disjunction = (
                len(possible_dirs) == 2
                and listed_live_dirs
                and listed_live_dirs != set(possible_dirs)
            )
            if incomplete_disjunction:
                # SE listed without SW: cannot pick SE alone.
                gold_keys = [
                    key for key in sorted(options)
                    if self.is_undetermined_option(options[key])
                ]
            elif listed_live and (
                len(possible_dirs) != 2
                or listed_live_dirs == set(possible_dirs)
            ):
                gold_keys = listed_live
            elif both_unknown:
                gold_keys = [
                    key for key in sorted(options)
                    if self.is_undetermined_option(options[key])
                ]
            else:
                gold_keys = [
                    key for key in sorted(options)
                    if self.is_none_of_above(options[key])
                ]
            valid = gold_keys
            none_gold = (not both_unknown) and not listed_live and not incomplete_disjunction
            for key in sorted(options):
                val = options[key]
                inn = key in gold_keys
                reason = self._type0_reason(
                    val, possible_dirs, both_unknown, none_gold,
                    incomplete_disjunction,
                    x_rel, y_rel, x_conflict, y_conflict,
                )
                verdicts.append((key, val, inn, reason))

        elif q_type == 1:
            m = re.search(
                r"Which object is in the (\w+) of ([^?]+?)\?",
                question_part,
            )
            if not m:
                return Grade(raw="Error: cannot parse type-1 question", error="parse type-1", q_type=1)
            direction = m.group(1).strip().title()
            obj_ref = _strip_the(m.group(2))
            axes = _DIR_AXES_MAP.get(direction, ())
            for key in sorted(options):
                candidate = _strip_the(options[key])
                val = options[key]
                if self.is_undetermined_option(candidate) or self.is_none_of_above(candidate):
                    continue
                if candidate not in objects_set:
                    verdicts.append((key, val, False, "not an entity in the map"))
                    continue
                inn = all(self._in_axis(x_reach, y_reach, candidate, obj_ref, ax) for ax in axes)
                reason = (
                    f"proven on every required axis of {direction}"
                    if inn else
                    f"not proven on every required axis of {direction}"
                )
                verdicts.append((key, val, inn, reason))
                if inn:
                    valid.append(key)
            if not valid:
                for key in sorted(options):
                    if self.is_none_of_above(options[key]):
                        valid.append(key)
                if not valid:
                    for key in sorted(options):
                        if self.is_undetermined_option(options[key]):
                            valid.append(key)
            seen = {v[0] for v in verdicts}
            gold_set = set(valid)
            for key in sorted(options):
                if key in seen:
                    continue
                val = options[key]
                inn = key in gold_set
                if self.is_none_of_above(val):
                    reason = (
                        "no listed entity is proven"
                        if inn else
                        "a listed entity is proven"
                    )
                elif self.is_undetermined_option(val):
                    reason = (
                        "no entity is proven; cannot be determined"
                        if inn else
                        "an entity is proven, so not undetermined"
                    )
                else:
                    reason = "not proven"
                verdicts.append((key, val, inn, reason))

        elif q_type == 2:
            m = re.search(
                r"How many objects are in the (\w+) of ([^?]+?)\?",
                question_part,
            )
            if not m:
                return Grade(raw="Error: cannot parse type-2 question", error="parse type-2", q_type=2)
            direction = m.group(1).strip().title()
            obj_ref = _strip_the(m.group(2))
            axes = _DIR_AXES_MAP.get(direction, ())
            count = sum(
                1
                for obj in objects
                if obj != obj_ref
                and all(self._in_axis(x_reach, y_reach, obj, obj_ref, ax) for ax in axes)
            )
            for key in sorted(options):
                val = options[key]
                if self.is_none_of_above(val) or self.is_undetermined_option(val):
                    continue
                try:
                    inn = int(str(val).strip()) == count
                    reason = (
                        f"matches the derived count {count}"
                        if inn else
                        f"count is {count}, not {val}"
                    )
                except (ValueError, TypeError):
                    inn = False
                    reason = "not a count"
                verdicts.append((key, val, inn, reason))
                if inn:
                    valid.append(key)
            if not valid:
                for key in sorted(options):
                    if self.is_none_of_above(options[key]):
                        valid.append(key)
            seen = {v[0] for v in verdicts}
            gold_set = set(valid)
            for key in sorted(options):
                if key in seen:
                    continue
                val = options[key]
                inn = key in gold_set
                if self.is_none_of_above(val):
                    reason = (
                        f"derived count {count} is not among A–D"
                        if inn else
                        f"derived count {count} is listed"
                    )
                else:
                    reason = "not a count"
                verdicts.append((key, val, inn, reason))

        else:
            return Grade(raw="Error: unknown question type", error="unknown type", q_type=q_type)

        raw = ",".join(valid) if valid else "No valid options found"
        return Grade(
            raw=raw,
            q_type=q_type,
            options=options,
            verdicts=verdicts,
            x_rel=x_rel,
            y_rel=y_rel,
            x_conflict=x_conflict,
            y_conflict=y_conflict,
        )

    def _type0_reason(
        self, val, possible_dirs, both_unknown, none_gold,
        incomplete_disjunction,
        x_rel, y_rel, x_conflict, y_conflict,
    ) -> str:
        if self.is_undetermined_option(val):
            if both_unknown:
                bits = []
                if x_conflict:
                    bits.append("X-axis contradiction")
                if y_conflict:
                    bits.append("Y-axis contradiction")
                if bits:
                    return " and ".join(bits) + ", so no compound is proven"
                return "neither axis is derived, so no compound is proven"
            if incomplete_disjunction:
                return (
                    "only one of the two remaining compounds is listed, "
                    "so the disjunction cannot be answered"
                )
            return "an axis is derived, so the direction is not fully unknown"
        if self.is_none_of_above(val):
            if none_gold:
                return "the proven direction is not listed"
            if both_unknown:
                return "no compound is proven; this is not a missing listed direction"
            return "a listed compound matches"
        if val in possible_dirs:
            if len(possible_dirs) == 1:
                return f"{val} both axes derived"
            return f"{val} remains possible given the derived axis"
        if both_unknown:
            if x_conflict or y_conflict:
                return "axis contradiction, so this direction is not proven"
            return "neither axis is derived, so this direction is not proven"
        return "does not match the derived axis"

    def agrees(self, text: str, answer: str) -> bool:
        """True iff ``answer`` (e.g. 'A, B') is exactly the v6 gold for ``text``."""
        grade = self.grade(text)
        if not grade.accept:
            return False
        return grade.letters == _letter_set(answer)


# Module-level wrappers so clean_v6 can `from spatial_solver import solve`.
_SOLVER = SpatialSolver()


def parse_problem(text: str):
    return _SOLVER.parse_problem(text)


def parse_options(question_part: str) -> dict[str, str]:
    return _SOLVER.parse_options(question_part)


def solve(text: str) -> str:
    return _SOLVER.solve(text)
