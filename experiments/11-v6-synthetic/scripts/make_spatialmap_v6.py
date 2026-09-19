#!/usr/bin/env python3
"""SpatialMap TQA with v6 fifth option and SpatialSolver gold.

Reads data/spatialeval_org.jsonl (fallback: spatialeval_cleaned.jsonl).
Appends one fifth option and re-grades with spatial/spatial_solver.py.

Writes data/spatialeval_v6_corr.jsonl (1500 rows).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "spatial"))
from spatial_solver import SpatialSolver  # noqa: E402

ORG = ROOT / "data" / "spatialeval_org.jsonl"
CLEAN = ROOT / "data" / "spatialeval_cleaned.jsonl"
DST = ROOT / "data" / "spatialeval_v6_corr.jsonl"
UNDET = "Cannot be determined"
NONE = "None of the Options"


def qtype(text: str) -> str:
    if "In which direction is" in text:
        return "dir"
    if "Which object is in the" in text:
        return "which"
    if "How many objects are in the" in text:
        return "count"
    return "?"


def attach_e(text: str, label: str) -> str:
    if re.search(rf"\bE\.\s*{re.escape(label)}", text, re.I):
        return text
    t = text.rstrip()
    if t.endswith("."):
        t = t[:-1]
    return t + f"\nE. {label}."


def main() -> None:
    src = ORG if ORG.is_file() else CLEAN
    if not src.is_file():
        raise SystemExit(f"missing {ORG} and {CLEAN}")
    solver = SpatialSolver()
    counts: Counter[str] = Counter()
    n = 0
    DST.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as fin, DST.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "")
            t = qtype(text)
            g0 = solver.grade(text)
            if g0.accept:
                fifth = UNDET if t == "dir" else NONE
            else:
                fifth = UNDET if t == "dir" else NONE
            text2 = attach_e(text, fifth)
            g1 = solver.grade(text2)
            row["text"] = text2
            if g1.accept:
                row["oracle_option"] = g1.raw
                inns = [val for _, val, inn, _ in g1.verdicts if inn]
                row["oracle_answer"] = inns[0] if inns else ""
                row["oracle_full_answer"] = " | ".join(
                    f"{let}. {val}" for let, val, inn, _ in g1.verdicts if inn
                )
                n_content = sum(
                    1
                    for let, val, inn, _ in g1.verdicts
                    if inn
                    and not solver.is_none_of_above(val)
                    and not solver.is_undetermined_option(val)
                )
                if n_content == 0:
                    counts[f"{t}-E"] += 1
                else:
                    counts[f"{t}-{n_content}"] += 1
            else:
                row["oracle_option"] = ""
                row["clean_note"] = g1.raw
                counts[f"{t}-empty"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {DST} n={n} from {src.name}")
    print("mix:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
