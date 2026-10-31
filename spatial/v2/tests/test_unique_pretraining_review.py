"""Shortcut features must not inspect labels, proofs or names."""

import importlib.util
from copy import deepcopy
from pathlib import Path

PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments/spatialentail-composition-v1/review_unique_pretraining.py"
)
SPEC = importlib.util.spec_from_file_location("pretraining_review", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def example():
    return {
        "problem": {
            "objects": ["Park", "Library", "Bank"],
            "premise": {
                "operands": [
                    {"subject": "Library", "reference": "Park", "allowed": ["North"]},
                    {"subject": "Bank", "reference": "Park", "allowed": ["East"]},
                ]
            },
        },
        "rows": [{"answer": "E", "references": {"natural": "secret"}}],
        "source": "targeted",
        "depth": 7,
    }


def test_labels_and_internal_metadata_cannot_change_features():
    record = example()
    expected = MODULE.features(record)
    record["rows"] = [{"answer": "F", "references": {"natural": "different"}}]
    record["source"] = "controlled"
    record["depth"] = 1
    assert MODULE.features(record) == expected


def test_renaming_and_premise_order_do_not_change_features():
    record = example()
    expected = MODULE.features(record)
    changed = deepcopy(record)
    names = {"Park": "P", "Library": "L", "Bank": "B"}
    changed["problem"]["objects"] = [
        names[x] for x in reversed(record["problem"]["objects"])
    ]
    for atom in changed["problem"]["premise"]["operands"]:
        atom["subject"] = names[atom["subject"]]
        atom["reference"] = names[atom["reference"]]
    changed["problem"]["premise"]["operands"].reverse()
    assert MODULE.features(changed) == expected
