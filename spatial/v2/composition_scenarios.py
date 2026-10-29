"""Select qualitative witness projections; numerical completions are audit-only."""

from itertools import groupby, pairwise

from spatial.v2.solver import Direction, direction_signs


def _order(coordinates, objects, axis):
    ordered = sorted(objects, key=lambda obj: (coordinates[obj][axis], obj))
    return [
        list(group)
        for _, group in groupby(ordered, key=lambda obj: coordinates[obj][axis])
    ]


def _project(model, problem):
    coordinates = {obj: (x, y) for obj, x, y in model["points"]}
    query = model["query_details"]
    reference = query["reference"]
    scenario = {
        "kind": "scenario",
        "query": model["query"],
        "outcome": model["outcome"],
        "reference": reference,
    }
    if model["query"] == "direction":
        target = query["target"]
        neighbors = {obj: set() for obj in (target, reference)}
        for atom in problem.premise.operands:
            for obj, adjacent in neighbors.items():
                if obj == atom.subject:
                    adjacent.add(atom.reference)
                elif obj == atom.reference:
                    adjacent.add(atom.subject)
        # Shared premise anchors make alternatives understandable, e.g. O < X < Y.
        objects = {target, reference} | (neighbors[target] & neighbors[reference])
        scenario.update(
            target=target,
            orders={
                axis: _order(coordinates, objects, i) for i, axis in enumerate("xy")
            },
        )
    else:
        relations = []
        for candidate in query["candidates"]:
            signs = tuple(
                (a > b) - (a < b)
                for a, b in zip(coordinates[candidate], coordinates[reference])
            )
            direction = next(d.value for d in Direction if direction_signs(d) == signs)
            relations.append([candidate, direction])
        scenario.update(
            relations=relations,
            required_direction=query["direction"],
            members=sorted(obj for obj, d in relations if d == query["direction"]),
        )
    return scenario


def qualitative_outline(units, problem, version):
    """Deduplicate displayed alternatives, retaining every original completion."""
    result, scenarios, audit = [], [], []
    recent = []
    for unit in units:
        if unit["kind"] == "model":
            projection = _project(unit, problem)
            index = next((i for i, s in enumerate(scenarios) if s == projection), None)
            if index is None:
                index = len(scenarios)
                scenarios.append(projection)
                result.append({**projection, "case": index + 1})
            audit.append({"case": index + 1, "points": unit["points"]})
            recent.append(index + 1)
        elif unit["kind"] == "membership" and unit["status"] == "contingent":
            if len(recent) != 2:
                raise ValueError("contingent membership requires two alternatives")
            result.append({**unit, "cases": recent})
            recent = []
        else:
            result.append(unit)
            recent = []
    return {"version": version, "units": result, "audit_witnesses": audit}


def render_scenario(unit, symbolic, join, abbreviations):
    label = f"Case {unit['case']}"
    if unit["query"] == "direction":
        parts = []
        for axis, groups in unit["orders"].items():
            if symbolic:
                parts.append(
                    " < ".join(
                        " = ".join(f"{axis}({obj})" for obj in group)
                        for group in groups
                    )
                    + "."
                )
            elif len(groups) == 1:
                parts.append(
                    f"{join(groups[0])} share the same {'horizontal position' if axis == 'x' else 'height'}."
                )
            elif all(len(group) == 1 for group in groups):
                parts.append(
                    f"From {'left to right' if axis == 'x' else 'bottom to top'}, the order is {join(group[0] for group in groups)}."
                )
            else:
                for lower, upper in pairwise(groups):
                    parts.append(
                        f"{join(lower)} {'lies' if len(lower) == 1 else 'lie'} "
                        f"{'to the left of' if axis == 'x' else 'below'} {join(upper)}."
                    )
                for group in groups:
                    if len(group) > 1:
                        parts.append(
                            f"{join(group)} share the same {'horizontal position' if axis == 'x' else 'height'}."
                        )
        outcome = (
            f"=> {abbreviations[unit['outcome']]}({unit['target']},{unit['reference']})."
            if symbolic
            else f"So {unit['target']} is {unit['outcome'].lower()} of {unit['reference']}."
        )
    else:
        grouped = {}
        for obj, direction in unit["relations"]:
            grouped.setdefault(direction, []).append(obj)
        parts = [
            (
                " & ".join(
                    f"{abbreviations[d]}({obj},{unit['reference']})" for obj in objects
                )
                + "."
                if symbolic
                else f"{join(objects)} {'is' if len(objects) == 1 else 'are'} {d.lower()} of {unit['reference']}."
            )
            for d, objects in grouped.items()
        ]
        members = unit["members"]
        outcome = (
            "Qualifying objects = {" + ", ".join(members) + "}."
            if symbolic
            else f"Only {join(members)} {'qualifies' if len(members) == 1 else 'qualify'} in this arrangement."
            if members
            else "No eligible object qualifies in this arrangement."
        )
        if unit["query"] == "count":
            outcome += (
                f" count={unit['outcome']}."
                if symbolic
                else f" This gives a count of {unit['outcome']}."
            )
    intro = (
        f"{label} (possible together under all premises): "
        if symbolic
        else f"{label}: One possibility consistent with all premises is the following. "
    )
    return intro + " ".join(parts) + " " + outcome
