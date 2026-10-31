"""Necessary reserve supply bounds must not turn unknowns into rejected supply."""

import importlib.util
import sys
from pathlib import Path

from spatial.v2.entailment_scene_protocol import all_scene_problem
from spatial.v2.tests.test_entailment_scene_protocol import scoped_problem

DIRECTORY = (
    Path(__file__).resolve().parents[3] / "experiments/spatialentail-composition-v1"
)
SPEC = importlib.util.spec_from_file_location(
    "fixed_reserve_diagnostic", DIRECTORY / "check_fixed_reserve_all_scene.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


def test_count_target_recomputed_over_all_scene_objects():
    problem = all_scene_problem(scoped_problem("count"))
    old = MODULE.check_count(("test", "train", 2, "family", 2, problem))
    new = MODULE.check_count(("test", "train", 2, "family", 3, problem))
    assert old["entailed_values"] == [3]
    assert new["entailed_values"] == [3]
    assert MODULE.shortage_table([old], {"train/count/2/3": 1}) == []
    assert new["consistent"]


def test_unresolved_smt_counts_as_optimistic_supply(monkeypatch):
    def inconclusive(*args, **kwargs):
        raise ValueError("inconclusive independent SMT audit")

    monkeypatch.setattr(MODULE, "checked_sat", inconclusive)
    problem = all_scene_problem(scoped_problem("count"))
    result = MODULE.check_count(("test", "train", 2, "family", 2, problem))
    assert result["status"] == "unresolved"
    assert MODULE.shortage_table([result], {"train/count/2/2": 1}) == []
    assert MODULE.shortage_table([result], {"train/count/2/2": 2}) == [
        {
            "cell": "train/count/2/2",
            "required": 2,
            "optimistic_supply": 1,
            "shortfall": 1,
        }
    ]


def test_nonentailed_target_cannot_fill_a_controlled_count_quota():
    result = {
        "stratum": "train/count/2",
        "target": 2,
        "entailed_values": [],
        "unresolved_values": [],
    }
    assert MODULE.shortage_table([result], {"train/count/2/2": 1})[0]["shortfall"] == 1
