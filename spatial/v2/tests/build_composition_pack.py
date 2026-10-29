"""Regenerate the fixture-based evidence feasibility pack, not training data.

Run with pytest and z3-solver installed because the source fixtures live in the
test suite. Rejections are reported explicitly and cause a nonzero exit status.
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from spatial.v2.composition_contracts import UnsupportedEvidence
from spatial.v2.composition_evidence import build_example
from spatial.v2.grading import AnswerMode
from spatial.v2.tests.test_pivot_evidence import BY_ID, VIEWS


def build_pack() -> dict:
    rows, rejected = [], []
    modes = (AnswerMode.SINGLE, AnswerMode.VISIBLE_POSSIBLE, AnswerMode.ALL_POSSIBLE)
    # Count and Selection questions about identical premises are one story.
    story_ids = {"C2": "S3", "C3": "S2"}
    for index, (name, menu, expected) in enumerate(VIEWS):
        story_id = story_ids.get(name, name)
        family_id = "I1-I2" if name in {"I1", "I2"} else story_id
        for mode, answer in zip(modes, expected):
            view_id = f"{name}-menu{index}-{mode.value}"
            try:
                row = build_example(
                    BY_ID[name].problem(),
                    menu,
                    mode,
                    story_id=story_id,
                    family_id=family_id,
                )
            except UnsupportedEvidence as exc:
                rejected.append({"view_id": view_id, "reason": str(exc)})
                continue
            if row["answer"] != answer:
                raise ValueError(
                    f"{view_id}: generated answer disagrees with fixture oracle"
                )
            row["view_id"] = view_id
            row["fixture_id"] = name
            rows.append(row)
    root = Path(__file__).resolve().parents[3]
    files = (
        "spatial/v2/composition_contracts.py",
        "spatial/v2/composition_conflicts.py",
        "spatial/v2/composition_evidence.py",
        "spatial/v2/composition_direction.py",
        "spatial/v2/solver.py",
        "spatial/v2/proofs.py",
        "spatial/v2/answer_certificates.py",
        "spatial/v2/which_certificates.py",
        "spatial/v2/model_certificates.py",
        "spatial/v2/certificate_generation.py",
        "spatial/v2/tests/test_pivot_evidence.py",
        "spatial/v2/tests/build_composition_pack.py",
    )
    return {
        "status": "fixture-feasibility-only-not-a-dataset-release",
        "fixture_cases": len(BY_ID),
        "unique_stories": len({row["story_id"] for row in rows}),
        "unique_families": len({row["family_id"] for row in rows}),
        "views": len(rows),
        "reference_renderings": 2 * len(rows),
        "solver_version": version("z3-solver"),
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in files
        },
        "rejected": rejected,
        "limits": [
            "Fixed hand-authored premises; generated evidence, not randomized story generation.",
            "Count support: at most eight candidates; fixed memberships and pair conflicts only.",
            "Higher-order Count correlations and unrenderable proofs are explicitly rejected.",
            "No dataset split, final renderer/token freeze, release audit, or training.",
            "No claim of minimum conflict subsets or arbitrary model-explanation verification.",
        ],
        "examples": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack = build_pack()
    args.output.write_text(json.dumps(pack, indent=2) + "\n")
    print(
        f"{pack['views']} views; {pack['unique_stories']} stories; "
        f"{pack['reference_renderings']} paired renderings; "
        f"{len(pack['rejected'])} rejected."
    )
    if pack["rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
