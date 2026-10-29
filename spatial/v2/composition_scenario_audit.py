"""Parse qualitative alternatives and check audit-only whole-map completions.

No renderer or projection helpers are imported. Printed claims, including joint
membership outcomes and case references, are checked against complete witnesses.
"""

import re
from itertools import combinations, pairwise

SIGNS = {
    "North": (0, 1),
    "Northeast": (1, 1),
    "East": (1, 0),
    "Southeast": (1, -1),
    "South": (0, -1),
    "Southwest": (-1, -1),
    "West": (-1, 0),
    "Northwest": (-1, 1),
}
SHORT = dict(zip(("N", "NE", "E", "SE", "S", "SW", "W", "NW"), SIGNS))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def names(text):
    return re.split(r", | and ", text)


def sign(a, b):
    return (a > b) - (a < b)


def _orders(text, symbolic):
    facts = []
    for sentence in text.split(". "):
        sentence = sentence.rstrip(".")
        if symbolic:
            require(
                re.fullmatch(r"[xy]\(\w+\)(?: [<=] [xy]\(\w+\))*", sentence),
                "invalid scenario order",
            )
            nodes = re.findall(r"([xy])\((\w+)\)", sentence)
            require(len({axis for axis, _ in nodes}) == 1, "mixed axes")
            signs = re.findall(r" ([<=]) ", sentence)
            facts.extend(
                (nodes[0][0], a[1], b[1], -1 if s == "<" else 0)
                for a, b, s in zip(nodes, nodes[1:], signs)
            )
            continue
        equal = re.fullmatch(
            r"(.+) share the same (horizontal position|height)", sentence
        )
        ordered = re.fullmatch(
            r"From (left to right|bottom to top), the order is (.+)", sentence
        )
        mixed = re.fullmatch(r"(.+) (?:lies|lie) (to the left of|below) (.+)", sentence)
        if equal:
            group = names(equal[1])
            facts.extend(
                ("x" if equal[2] == "horizontal position" else "y", a, b, 0)
                for a, b in pairwise(group)
            )
        elif ordered:
            group = names(ordered[2])
            facts.extend(
                ("x" if ordered[1] == "left to right" else "y", a, b, -1)
                for a, b in pairwise(group)
            )
        else:
            require(mixed is not None, "invalid qualitative order sentence")
            facts.extend(
                ("x" if mixed[2] == "to the left of" else "y", a, b, -1)
                for a in names(mixed[1])
                for b in names(mixed[3])
            )
    return facts


def _closure(facts):
    """Normalize printed equality/strict-order chains without trusting witnesses."""
    normalized = []
    for axis in "xy":
        objects = sorted({obj for ax, a, b, _ in facts if ax == axis for obj in (a, b)})
        edges = {(obj, obj): False for obj in objects}
        for ax, a, b, s in facts:
            if ax != axis:
                continue
            edges[a, b] = edges.get((a, b), False) or s == -1
            if s == 0:
                edges[b, a] = edges.get((b, a), False)
        for k in objects:
            for a in objects:
                for b in objects:
                    if (a, k) in edges and (k, b) in edges:
                        edges[a, b] = (
                            edges.get((a, b), False) or edges[a, k] or edges[k, b]
                        )
        require(
            not any(edges[obj, obj] for obj in objects),
            "inconsistent printed scenario order",
        )
        for a, b in combinations(objects, 2):
            require(
                (a, b) in edges or (b, a) in edges, "incomplete printed scenario order"
            )
            normalized.append(
                (axis, a, b, -1 if edges.get((a, b)) else 1 if edges.get((b, a)) else 0)
            )
    return normalized


def _parse(paragraph, symbolic):
    match = re.fullmatch(
        r"Case (\d+) \(possible together under all premises\): (.+)"
        if symbolic
        else r"Case (\d+): One possibility consistent with all premises is the following\. (.+)",
        paragraph,
    )
    require(match is not None, "unrecognized scenario paragraph")
    number, body = int(match[1]), match[2]
    if (
        "Qualifying objects = " in body
        or "qualifies in this arrangement." in body
        or "qualify in this arrangement." in body
    ):
        if symbolic:
            relations_text, outcome_text = body.split(" Qualifying objects = ", 1)
            match = re.fullmatch(r"\{([^}]*)\}\.(?: count=(\d+)\.)?", outcome_text)
            require(match is not None, "invalid scenario membership outcome")
            members = match[1].split(", ") if match[1] else []
            count = int(match[2]) if match[2] is not None else None
            relations = []
            for clause in re.split(r"\. | & ", relations_text.rstrip(".")):
                m = re.fullmatch(r"(\w+)\((\w+),(\w+)\)", clause)
                require(
                    m is not None and m[1] in SHORT, "invalid qualitative predicate"
                )
                relations.append((m[2], m[3], SHORT[m[1]]))
        else:
            relations_text, outcome_text = re.split(
                r" (?=Only |No eligible object )", body, maxsplit=1
            )
            match = re.fullmatch(
                r"(?:Only (.+) (?:qualifies|qualify)|No eligible object qualifies) in this arrangement\.(?: This gives a count of (\d+)\.)?",
                outcome_text,
            )
            require(match is not None, "invalid scenario membership sentence")
            members = names(match[1]) if match[1] else []
            count = int(match[2]) if match[2] is not None else None
            relations = []
            for clause in relations_text.rstrip(".").split(". "):
                m = re.fullmatch(r"(.+) (?:is|are) (\w+) of (\w+)", clause)
                require(
                    m is not None and m[2].capitalize() in SIGNS,
                    "invalid spatial alternative",
                )
                relations.extend((obj, m[3], m[2].capitalize()) for obj in names(m[1]))
        return number, {
            "relations": sorted(relations),
            "members": sorted(members),
            "count": count,
        }
    if symbolic:
        order_text, outcome_text = body.split(" => ", 1)
        match = re.fullmatch(r"(\w+)\((\w+),(\w+)\)\.", outcome_text)
        require(match is not None and match[1] in SHORT, "invalid direction outcome")
        target, reference, direction = match[2], match[3], SHORT[match[1]]
    else:
        order_text, outcome_text = body.split(" So ", 1)
        match = re.fullmatch(r"(\w+) is (\w+) of (\w+)\.", outcome_text)
        require(
            match is not None and match[2].capitalize() in SIGNS,
            "invalid direction outcome",
        )
        target, reference, direction = match[1], match[3], match[2].capitalize()
    return number, {
        "orders": _closure(_orders(order_text, symbolic)),
        "target": target,
        "reference": reference,
        "direction": direction,
    }


def check_scenarios(row, problem):
    outline = row["explanation_outline"]
    expected = {u["case"]: u for u in outline["units"] if u["kind"] == "scenario"}
    parsed = {}
    for style, text in row["references"].items():
        require(
            not re.search(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", text),
            "numerical coordinates in supervision",
        )
        cases = {}
        for paragraph in text.split("\n\n"):
            if paragraph.startswith("Case "):
                number, case = _parse(paragraph, style == "symbolic")
                require(number not in cases, "duplicate scenario description")
                cases[number] = case
            elif "qualifies in Case " in paragraph:
                m = re.match(
                    r"(\w+) qualifies in Case (\d+)(?:, not| but not) Case (\d+)",
                    paragraph,
                )
                require(m is not None, "invalid scenario cross-reference")
                obj, first, second = m[1], int(m[2]), int(m[3])
                require(
                    first in cases and second in cases,
                    "unintroduced scenario reference",
                )
                require(
                    obj in cases[first]["members"]
                    and obj not in cases[second]["members"],
                    "wrong contingency case references",
                )
        require(set(cases) == set(expected), "missing or extra scenario")
        parsed[style] = cases
    require(
        parsed["natural"] == parsed["symbolic"],
        "different qualitative evidence across styles",
    )
    cases = parsed["natural"]
    witnessed = set()
    for witness in outline["audit_witnesses"]:
        number = witness["case"]
        require(number in cases, "orphan audit witness")
        case, unit = cases[number], expected[number]
        coords = {obj: (x, y) for obj, x, y in witness["points"]}
        require(
            len(coords) == len(witness["points"])
            and set(coords) == set(problem["objects"]),
            "missing or duplicate audit objects",
        )
        require(len(set(coords.values())) == len(coords), "coincident audit objects")

        def relation(a, b, coords=coords):
            return tuple(sign(x, y) for x, y in zip(coords[a], coords[b]))

        for atom in problem["premise"]["operands"]:
            require(
                relation(atom["subject"], atom["reference"])
                == SIGNS[atom["allowed"][0]],
                "audit completion violates premise",
            )
        if "orders" in case:
            objects = {
                obj
                for groups in unit["orders"].values()
                for group in groups
                for obj in group
            }
            for axis in "xy":
                require(
                    {
                        obj
                        for ax, a, b, _ in case["orders"]
                        if ax == axis
                        for obj in (a, b)
                    }
                    == objects,
                    "missing scenario order objects",
                )
            for axis, a, b, s in case["orders"]:
                require(
                    relation(a, b)["xy".index(axis)] == s,
                    "printed alternative contradicts audit completion",
                )
            require(
                case["target"] == problem["query"]["target"]
                and case["reference"] == problem["query"]["reference"],
                "wrong scenario query",
            )
            require(
                relation(case["target"], case["reference"]) == SIGNS[case["direction"]],
                "wrong scenario direction",
            )
            # Both printed axes must themselves establish the stated query outcome.
            for i, axis in enumerate("xy"):
                a, b = sorted((case["target"], case["reference"]))
                value = next(
                    s for ax, x, y, s in case["orders"] if (ax, x, y) == (axis, a, b)
                )
                require(
                    value * (1 if a == case["target"] else -1)
                    == SIGNS[case["direction"]][i],
                    "orders do not establish outcome",
                )
        else:
            candidates = problem["query"]["candidates"]
            require(
                sorted(obj for obj, _, _ in case["relations"]) == sorted(candidates),
                "missing or duplicated candidate alternatives",
            )
            for obj, reference, direction in case["relations"]:
                require(
                    reference == problem["query"]["reference"]
                    and relation(obj, reference) == SIGNS[direction],
                    "printed alternative contradicts audit completion",
                )
            members = sorted(
                obj
                for obj, _, d in case["relations"]
                if d == problem["query"]["directions"][0]
            )
            require(case["members"] == members, "wrong qualitative membership outcome")
            if unit["query"] == "count":
                require(case["count"] == len(members), "wrong joint count")
            else:
                require(case["count"] is None, "unexpected count")
        witnessed.add(number)
    require(witnessed == set(cases), "scenario lacks a valid completion")
    return len(cases)
