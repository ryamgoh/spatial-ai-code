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
