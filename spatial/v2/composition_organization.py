"""Organize verified evidence without inventing a search trace or new deductions."""

import json
from copy import deepcopy

from spatial.v2.composition_shared_paths import factor_shared_paths
from spatial.v2.solver import DirectionQuery, WhichQuery

VERSION = "readable-question-led-v5"


def evidence_units(outline):
    """Remove presentation-only additions for an exact evidence comparison."""
    units = []
    for original in outline["units"]:
        unit = deepcopy(original)
        if unit["kind"] == "purpose" or "comparison" in unit:
            continue
        unit.pop("shared_segments", None)
        if unit["kind"] == "membership-group":
            memberships = unit["memberships"]
            count = len(memberships)
            paths = units[-count:] if count else []
            if (
                len(paths) != count
                or count < 2
                or any(p["kind"] != "path" for p in paths)
            ):
                raise ValueError("grouped memberships lack their individual paths")
            units[-count:] = [item for pair in zip(paths, memberships) for item in pair]
        else:
            units.append(unit)
    return units


def evidence_inventory(outline):
    return sorted(json.dumps(u, sort_keys=True) for u in evidence_units(outline))


def _exclusion_key(paths, membership):
    if membership["status"] != "impossible" or len(paths) != 1:
        return None
    path = paths[0]
    candidate, reference = membership["candidate"], membership["reference"]
    a, b = path["nodes"][0], path["nodes"][-1]
    if {a, b} != {candidate, reference}:
        return None
    sign = (-1 if a == candidate else 1) if any(path["signs"]) else 0
    return path["axis"], sign, membership["direction"], reference


def _organize_run(blocks):
    result, groups = [], {}
    for paths, membership in blocks:
        key = _exclusion_key(paths, membership)
        if key is None:
            result.extend([*paths, membership])
        else:
            groups.setdefault(key, []).append((paths, membership))
    for (axis, sign, direction, reference), group in groups.items():
        routes = {}
        for paths, membership in group:
            path = paths[0]
            route = (
                tuple(
                    "candidate" if n == membership["candidate"] else n
                    for n in path["nodes"]
                ),
                tuple(path["signs"]),
            )
            routes.setdefault(route, []).append((paths, membership))
        group = [block for blocks in routes.values() for block in blocks]
        for paths, _ in group:
            result.extend(paths)
        if len(group) == 1:
            result.append(group[0][1])
        else:
            result.append(
                {
                    "kind": "membership-group",
                    "axis": axis,
                    "sign": sign,
                    "direction": direction,
                    "reference": reference,
                    "memberships": [membership for _, membership in group],
                }
            )
    return result


def organize_outline(outline, problem, mode):
    """Group only unconditional path-backed memberships; retain scenario scopes."""
    result = deepcopy(outline)
    units = evidence_units(result)
    organized, index = [], 0
    while index < len(units):
        blocks = []
        while index < len(units):
            stop = index
            while stop < len(units) and units[stop]["kind"] == "path":
                stop += 1
            if (
                stop == index
                or stop == len(units)
                or units[stop]["kind"] != "membership"
                or units[stop]["status"] not in ("entailed", "impossible")
            ):
                break
            blocks.append((units[index:stop], units[stop]))
            index = stop + 1
        if blocks:
            organized.extend(_organize_run(blocks))
        else:
            organized.append(units[index])
            index += 1
    query = problem.query
    purpose = {
        "kind": "purpose",
        "query_kind": "direction"
        if isinstance(query, DirectionQuery)
        else "which"
        if isinstance(query, WhichQuery)
        else "count",
        "reference": query.reference,
        "mode": mode.value,
    }
    if hasattr(query, "target"):
        purpose["target"] = query.target
    else:
        purpose["direction"] = next(iter(query.directions)).value
    result["version"] = VERSION
    result["units"] = [purpose, *organized]
    if evidence_inventory(result) != evidence_inventory(outline):
        raise ValueError("organization changed selected evidence")
    return factor_shared_paths(result)
