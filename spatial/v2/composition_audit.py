"""Align both rendered styles into factual lines for auditable paired review.

This checks template fidelity, not arbitrary model explanations. It does not
replace solver/certificate checks or automatically issue inspection verdicts.
"""

import argparse
import difflib
import json
import re
from pathlib import Path

from spatial.v2.composition_training import audit_selection, content_hash


def normalize_natural(text):
    relations = {"equal to": "=", "less than": "<", "greater than": ">"}
    text = re.sub(
        r"(\w+)'s ([xy])-coordinate (?:is )?(equal to|less than|greater than) (\w+)'s \2-coordinate",
        lambda m: f"{m[2]}({m[1]}) {relations[m[3]]} {m[2]}({m[4]})",
        text,
    )
    text = re.sub(
        r"(\w+) is exactly (north|northeast|east|southeast|south|southwest|west|northwest) of (\w+)",
        lambda m: f"{m[2].capitalize()}({m[1]},{m[3]})",
        text,
    )
    lines = []
    for line in text.splitlines():
        if line.startswith("A valid map places "):
            points, outcome = line[len("A valid map places ") :].split(
                ". In this map, "
            )
            points = re.sub(
                r"(\w+) at \((-?\d+), (-?\d+)\)", r"\1:(\2,\3)", points
            ).replace("; ", ", ")
            line = f"Valid map {{{points}}} => {outcome}"
        elif line.startswith("Composing this path gives "):
            line = line.replace("Composing this path gives ", "Transitivity: ", 1)
        elif re.match(r"From p\d+, ", line):
            line = re.sub(r"From (p\d+), ", r"\1: ", line)
        elif ": From " in line:
            line = re.sub(r"^([^:]+): From (.*), (.*)\.$", r"\1: [\2] => \3", line)
        lines.append(line.rstrip("."))
    return "\n".join(lines)


def aligned_pair(row):
    natural = normalize_natural(row["references"]["natural"])
    symbolic = "\n".join(
        line.rstrip(".") for line in row["references"]["symbolic"].splitlines()
    )
    if natural != symbolic:
        difference = "\n".join(
            difflib.unified_diff(natural.splitlines(), symbolic.splitlines())
        )
        raise ValueError(f"paired rendered facts differ for {row['id']}:\n{difference}")
    return symbolic


def inspect_pack(pack):
    selection = audit_selection(pack)
    by_id = {r["id"]: r for r in pack["rows"]}
    for row in pack["rows"]:
        aligned_pair(row)
    return selection, by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    pack = json.loads((args.candidate / "pilot.json").read_text())
    selection, rows = inspect_pack(pack)
    print(
        f"All {len(rows)} pairs align. {len(selection['sample_ids'])} selected for inspection."
    )
    for name in selection["sample_ids"][args.offset : args.offset + args.count]:
        row = rows[name]
        print(f"\n## {name} SHA256 {content_hash(row)}")
        print(row["prompt"].split("Objects: ", 1)[1])
        print("ALIGNED NATURAL/SYMBOLIC FACTS:")
        text = aligned_pair(row)
        if args.inventory:
            # Inspect the contract-relevant conclusions and complete witnesses;
            # step validity and full paired wording are checked separately.
            text = "\n".join(
                line
                for line in text.splitlines()
                if line.startswith(
                    (
                        "Valid map",
                        "Q-DIR:",
                        "Transitivity:",
                        "The ",
                        "Answer:",
                        "No ",
                        "Multiple ",
                        "These ",
                        "Count ",
                        "Fixed membership",
                        "Thus ",
                        "Conflicting",
                        "Suppose both",
                        "A strict",
                    )
                )
                or re.fullmatch(r"\w+: (entailed|contingent|impossible)", line)
                or "violating the distinct-point" in line
            )
        print(text)


if __name__ == "__main__":
    main()
