"""Factor unconditional repeated axis paths without changing their evidence."""

from collections import defaultdict
from copy import deepcopy


def _key(path, start, stop):
    nodes = tuple(path["nodes"][start : stop + 1])
    signs = tuple(path["signs"][start:stop])
    indices = tuple(path["indices"][start:stop])
    if not any(signs) and nodes[::-1] < nodes:
        nodes, indices = nodes[::-1], indices[::-1]
    return path["axis"], nodes, signs, indices, path["original_count"]


def factor_shared_paths(outline):
    """Greedily replace disjoint repeated subpaths; never hoist assumptions."""
    result = deepcopy(outline)
    units = result["units"]
    if any(u["kind"] in ("assume-direction", "assume-pair", "conflict") for u in units):
        return result
    paths = {
        i: u
        for i, u in enumerate(units)
        if u["kind"] == "path"
        and len(set(u["nodes"])) == len(u["nodes"])
        and all(index < u["original_count"] for index in u["indices"])
    }
    candidates = defaultdict(list)
    for i, path in paths.items():
        for start in range(len(path["indices"])):
            for stop in range(start + 2, len(path["indices"]) + 1):
                candidates[_key(path, start, stop)].append((i, start, stop))
    occupied = defaultdict(set)
    definitions = defaultdict(list)
    label = 0
    while candidates:
        available = []
        for key, occurrences in candidates.items():
            free = [
                (i, start, stop)
                for i, start, stop in occurrences
                if not occupied[i].intersection(range(start, stop))
            ]
            length = len(key[3])
            saving = length * (len(free) - 1) - len(free)
            if len(free) >= 2 and saving > 0:
                available.append((saving, length, key, free))
        if not available:
            break
        _, _, key, occurrences = min(
            available, key=lambda item: (-item[0], -item[1], item[2])
        )
        del candidates[key]
        label += 1
        axis, nodes, signs, indices, original = key
        definition = {
            "kind": "path",
            "axis": axis,
            "nodes": list(nodes),
            "signs": list(signs),
            "indices": list(indices),
            "original_count": original,
            "comparison": label,
        }
        definitions[min(i for i, _, _ in occurrences)].append(definition)
        for i, start, stop in occurrences:
            occupied[i].update(range(start, stop))
            paths[i].setdefault("shared_segments", []).append(
                {"start": start, "stop": stop, "comparison": label}
            )
    # Number comparisons in first-use order, not by the greedy savings order.
    labels = {
        u["comparison"]: index
        for index, u in enumerate(
            (u for i in sorted(definitions) for u in definitions[i]), 1
        )
    }
    result["units"] = []
    for i, unit in enumerate(units):
        for definition in definitions[i]:
            definition["comparison"] = labels[definition["comparison"]]
            result["units"].append(definition)
        for segment in unit.get("shared_segments", []):
            segment["comparison"] = labels[segment["comparison"]]
        result["units"].append(unit)
    return result


def collapsed_path(unit):
    """Return printable links; full original links remain in the audit outline."""
    segments = {part["start"]: part for part in unit.get("shared_segments", [])}
    nodes, signs, citations = [unit["nodes"][0]], [], []
    index = 0
    while index < len(unit["indices"]):
        segment = segments.get(index)
        stop = segment["stop"] if segment else index + 1
        signs.append(-1 if any(unit["signs"][index:stop]) else 0)
        nodes.append(unit["nodes"][stop])
        citations.append(
            f"comparison {segment['comparison']}" if segment else unit["indices"][index]
        )
        index = stop
    return {**unit, "nodes": nodes, "signs": signs, "indices": citations}
