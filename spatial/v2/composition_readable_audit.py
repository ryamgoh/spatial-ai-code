"""Independent checks of printed axis links and witness maps in readable pairs.

This deliberately does not import the renderer or its direction-sign helper.
It checks these evidence-bearing text fragments, not arbitrary prose entailment.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from spatial.v2.composition_scenario_audit import check_scenarios

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
POINT = r"([A-Za-z]\w*)"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _names(text):
    return re.split(r", | and ", text)


def _chain(paragraph, style):
    if style == "symbolic":
        match = re.match(r"^\[([^]]+)\] (.+)\.$", paragraph)
        if not match:
            return None
        citations, body = match.groups()
        body, *endpoint = body.split(" => ")
        points = re.findall(rf"([xy])\({POINT}\)", body)
        signs = re.findall(r" ([<=]) ", body)
        require(len(points) == len(signs) + 1, "malformed printed chain")
        require(len({p[0] for p in points}) == 1, "printed chain mixes axes")
        axis = points[0][0]
        nodes = [p[1] for p in points]
        if endpoint:
            expected = (
                f"{axis}({nodes[0]}) {'<' if '<' in signs else '='} {axis}({nodes[-1]})"
            )
            require(endpoint == [expected], "incorrect printed endpoint")
        return axis, nodes, signs, citations.split(", ")
    match = re.search(r"\((?:premise )?sources: ([^)]+)\)", paragraph)
    if not match:
        return None
    citations = _names(match[1])
    body = paragraph[: match.start()].strip()
    equal = re.fullmatch(r"(.+) share the same ([xy])-coordinate", body)
    strict = re.fullmatch(r"The ([xy])-coordinates increase strictly along (.+)", body)
    single = re.fullmatch(
        rf"{POINT} has a greater ([xy])-coordinate than {POINT}", body
    )
    if equal:
        nodes, axis = _names(equal[1]), equal[2]
        signs = ["="] * (len(nodes) - 1)
    elif strict:
        axis, nodes = strict[1], _names(strict[2])
        signs = ["<"] * (len(nodes) - 1)
    elif single:
        axis, nodes, signs = single[2], [single[3], single[1]], ["<"]
    else:
        body = re.sub(
            r"^Following the [xy]-coordinates, |^The cited relations would require that ",
            "",
            body,
        )
        nodes, signs, axes = [], [], []
        for clause in body.split("; "):
            comparison = re.fullmatch(
                rf"{POINT} (shares its ([xy])-coordinate with|has a smaller ([xy])-coordinate than) {POINT}",
                clause,
            )
            require(comparison is not None, "unrecognized printed comparison")
            a, _relation, equal_axis, strict_axis, b = comparison.groups()
            require(not nodes or nodes[-1] == a, "disconnected printed chain")
            if not nodes:
                nodes.append(a)
            nodes.append(b)
            signs.append("=" if equal_axis else "<")
            axes.append(equal_axis or strict_axis)
        require(len(set(axes)) == 1, "printed chain mixes axes")
        axis = axes[0]
    suffix = paragraph[match.end() :]
    if "So " in suffix or "Therefore, " in suffix:
        require(
            suffix
            in (
                f". So {nodes[-1]} has a greater {axis}-coordinate than {nodes[0]}.",
                f". Therefore, {nodes[-1]} has a greater {axis}-coordinate than {nodes[0]}.",
            )
            and "<" in signs,
            "incorrect natural-language endpoint",
        )
    return axis, nodes, signs, citations


def _check_chain(chain, atoms, assumptions, comparisons=None):
    axis, nodes, signs, citations = chain
    require(len(signs) == len(citations), "citation/link count mismatch")
    normalized, expanded_nodes, expanded_signs = [], [nodes[0]], []
    for a, b, sign, citation in zip(nodes, nodes[1:], signs, citations):
        reference = re.fullmatch(r"comparison (\d+)", citation)
        if reference:
            label = int(reference[1])
            require(label in (comparisons or {}), "unknown or forward comparison")
            ref_axis, ref_nodes, ref_signs, ref_sources = comparisons[label]
            require(axis == ref_axis, "comparison uses wrong axis")
            if (a, b) == (ref_nodes[-1], ref_nodes[0]):
                require("<" not in ref_signs, "cannot reverse strict comparison")
                ref_nodes, ref_signs, ref_sources = (
                    ref_nodes[::-1],
                    ref_signs[::-1],
                    ref_sources[::-1],
                )
            require(
                (a, b) == (ref_nodes[0], ref_nodes[-1]), "wrong comparison endpoints"
            )
            require(sign == ("<" if "<" in ref_signs else "="), "wrong comparison sign")
            expanded_nodes.extend(ref_nodes[1:])
            expanded_signs.extend(ref_signs)
            normalized.extend(ref_sources)
            continue
        assumption = re.fullmatch(r"(?:temporary )?assumption (\d+)", citation)
        if assumption:
            index = int(assumption[1]) - 1
            require(0 <= index < len(assumptions), "unknown temporary assumption")
            atom = assumptions[index]
            normalized.append(("assumption", index))
        else:
            require(re.fullmatch(r"p?\d+", citation) is not None, "invalid citation")
            index = int(citation.removeprefix("p")) - 1
            require(0 <= index < len(atoms), "out-of-range citation")
            atom = atoms[index]
            normalized.append(("premise", index))
        require(len(atom["allowed"]) == 1, "non-atomic cited relation")
        expected = SIGNS[atom["allowed"][0]]["xy".index(axis)]
        if (a, b) == (atom["reference"], atom["subject"]):
            expected = -expected
        else:
            require(
                (a, b) == (atom["subject"], atom["reference"]), "wrong cited objects"
            )
        require(
            expected == (0 if sign == "=" else -1),
            "printed link contradicts cited premise",
        )
        expanded_nodes.append(b)
        expanded_signs.append(sign)
    return axis, expanded_nodes, expanded_signs, normalized


def _check_map(paragraph, style, problem):
    prefix = (
        "Example valid map satisfying all premises: "
        if style == "symbolic"
        else "For example, one valid arrangement consistent with all premises places "
    )
    if not paragraph.startswith(prefix):
        return None
    points_text, outcome_text = paragraph[len(prefix) :].split(". ", 1)
    pattern = (
        rf"{POINT}=\((-?\d+),(-?\d+)\)"
        if style == "symbolic"
        else rf"{POINT} at \((-?\d+), (-?\d+)\)"
    )
    entries = re.findall(pattern, points_text)
    coordinates = {obj: (int(x), int(y)) for obj, x, y in entries}
    require(
        len(entries) == len(coordinates) == len(problem["objects"]),
        "missing/duplicate map coordinates",
    )
    require(set(coordinates) == set(problem["objects"]), "map objects differ")
    require(
        len(set(coordinates.values())) == len(coordinates), "map has coincident objects"
    )

    def signs(a, b):
        return tuple((x > y) - (x < y) for x, y in zip(coordinates[a], coordinates[b]))

    for atom in problem["premise"]["operands"]:
        require(
            signs(atom["subject"], atom["reference"]) == SIGNS[atom["allowed"][0]],
            "printed map violates premise",
        )
    query = problem["query"]
    if "target" in query:
        outcome = next(
            d
            for d, s in SIGNS.items()
            if s == signs(query["target"], query["reference"])
        )
        expected = (
            f"direction={outcome}."
            if style == "symbolic"
            else f"This gives the direction {outcome.lower()}."
        )
        require(outcome_text == expected, "wrong printed map direction")
    else:
        members = sorted(
            obj
            for obj in query["candidates"]
            if signs(obj, query["reference"]) == SIGNS[query["directions"][0]]
        )
        if outcome_text.startswith(("count=", "This gives a count of ")):
            expected = (
                f"count={len(members)}."
                if style == "symbolic"
                else f"This gives a count of {len(members)}."
            )
            require(outcome_text == expected, "wrong joint count in printed map")
        elif style == "symbolic":
            require(
                outcome_text == "Qualifying objects = {" + ", ".join(members) + "}.",
                "wrong printed membership set",
            )
        else:
            names = (
                members[0]
                if len(members) == 1
                else ", ".join(members[:-1]) + " and " + members[-1]
                if members
                else ""
            )
            expected = (
                f"In this arrangement, only {names} {'qualifies' if len(members) == 1 else 'qualify'}."
                if members
                else "In this arrangement, no eligible objects qualify."
            )
            require(outcome_text == expected, "wrong printed membership set")
    return coordinates


def check_row(row, problem):
    qualitative = "audit_witnesses" in row["explanation_outline"]
    scenario_count = check_scenarios(row, problem) if qualitative else 0
    atoms = problem["premise"]["operands"]
    parsed = {}
    for style, text in row["references"].items():
        chains, maps, assumptions = [], [], []
        comparisons = {}
        labels = []
        for paragraph in text.split("\n\n"):
            definition = re.match(r"^Comparison (\d+): ", paragraph)
            if definition:
                paragraph = paragraph[definition.end() :]
            pair = re.fullmatch(
                rf"(?:Assume|Suppose) {POINT} and {POINT} both qualify \(temporary assumptions 1 and 2\)\.",
                paragraph,
            )
            single = (
                re.fullmatch(
                    rf"Assume (\w+)\({POINT},{POINT}\) \(temporary assumption 1\)\.",
                    paragraph,
                )
                if style == "symbolic"
                else re.fullmatch(
                    rf"Suppose {POINT} is exactly (\w+) of {POINT} \(temporary assumption 1\)\.",
                    paragraph,
                )
            )
            if pair:
                assumptions = [
                    {
                        "subject": obj,
                        "reference": problem["query"]["reference"],
                        "allowed": problem["query"]["directions"],
                    }
                    for obj in pair.groups()
                ]
            elif single:
                abbreviations = dict(
                    zip(("N", "NE", "E", "SE", "S", "SW", "W", "NW"), SIGNS)
                )
                direction, subject, reference = (
                    single.groups()
                    if style == "symbolic"
                    else (single[2], single[1], single[3])
                )
                direction = (
                    abbreviations[direction]
                    if style == "symbolic"
                    else direction.capitalize()
                )
                assumptions = [
                    {"subject": subject, "reference": reference, "allowed": [direction]}
                ]
            chain = _chain(paragraph, style)
            if chain:
                checked = _check_chain(chain, atoms, assumptions, comparisons)
                label = int(definition[1]) if definition else None
                if label is not None:
                    require(
                        label == len(comparisons) + 1,
                        "duplicate or unordered comparison",
                    )
                    require(
                        all(source[0] == "premise" for source in checked[3]),
                        "shared comparison depends on temporary assumption",
                    )
                    comparisons[label] = checked
                chains.append(checked)
                labels.append(label)
            require(
                not definition or chain is not None, "missing comparison definition"
            )
            model = _check_map(paragraph, style, problem)
            if model is not None:
                maps.append(model)
        units = row["explanation_outline"]["units"]
        require(
            len(chains) == sum(u["kind"] == "path" for u in units),
            "missing printed path",
        )
        for checked, label, unit in zip(
            chains, labels, (u for u in units if u["kind"] == "path")
        ):
            expected_sources = [
                ("premise", i)
                if i < unit["original_count"]
                else ("assumption", i - unit["original_count"])
                for i in unit["indices"]
            ]
            require(
                checked
                == (
                    unit["axis"],
                    unit["nodes"],
                    ["<" if s else "=" for s in unit["signs"]],
                    expected_sources,
                )
                and label == unit.get("comparison"),
                "printed path differs from selected evidence",
            )
        require(
            len(maps) == sum(u["kind"] == "model" for u in units), "missing printed map"
        )
        parsed[style] = (chains, maps)
    require(
        parsed["natural"] == parsed["symbolic"],
        "styles expose different printed chains or maps",
    )
    return {
        "paths_per_style": len(parsed["natural"][0]),
        "maps_per_style": len(parsed["natural"][1]),
        "qualitative_scenarios_per_style": scenario_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite text audit")
    totals = Counter()
    records = [json.loads(line) for line in args.source.read_text().splitlines()]
    for record in records:
        require(record["status"] == "accepted", "audit requires every base accepted")
        for row in record["data"]["rows"]:
            try:
                totals.update(check_row(row, record["problem"]))
            except (ValueError, KeyError, StopIteration) as exc:
                raise ValueError(f"{row['id']}: {exc}") from exc
            totals["pairs"] += 1
    result = {
        "status": "passed",
        "checks": dict(totals),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Printed chains, citations, endpoints, qualitative alternatives, cross-references and joint outcomes; audit-only whole-map completions; cross-style equality. Not a parser of all conclusion prose or a fluency judgment.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
