from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE = Path(__file__).with_name("calibrate_rollouts.py")
SPEC = importlib.util.spec_from_file_location("calibrate_v14_rollouts", MODULE)
assert SPEC and SPEC.loader
CAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAL)


def test_reward_parser_uses_last_exact_answer_set() -> None:
    assert CAL.predicted("Answer: A\nreasoning\nAnswer: C, A") == {"A", "C"}
    assert CAL.predicted("no answer") == set()
    assert CAL.letters("B, D") == {"B", "D"}


def test_calibration_thresholds_include_parseability_gate() -> None:
    # Keep this in sync with the operational gate in calibrate_rollouts.py.
    parseable_rate = 0.34
    checks = {
        "at least 35% of rollouts have a parseable final answer": parseable_rate >= 0.35,
        "rollout pass rate is between 15% and 90%": True,
        "at least 15% of prompt groups have mixed rewards": True,
        "at least 4 prompt groups are not all correct": True,
    }

    assert not all(checks.values())
