"""All-controlled plumbing must preserve the agreed numerical study controls."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from spatial.v2.entailment_replacement_allocation import (
    CONSTRUCTION_CANDIDATES_PER_SLOT,
    policy_from,
    schedule,
    validate_schedule,
)

EXP = Path(__file__).resolve().parents[3] / "experiments/spatialentail-composition-v1"


@pytest.fixture(scope="module")
def policy():
    return policy_from(
        json.loads((EXP / "entailment-config-frozen-v1.json").read_text())
    )


def test_preserves_exact_answer_depth_count_and_family_targets(policy):
    planned = schedule(policy)
    assert len(planned) == len({s.index for s in planned}) == 6000
    assert CONSTRUCTION_CANDIDATES_PER_SLOT * len(planned) == 18000
    assert set(policy["depth_family_minima"].values()) == {50}
    for kind in ("direction", "which", "count"):
        counts = Counter(s.category for s in planned if s.kind == kind)
        assert counts["A-D"] == 1200
        assert counts["unique-unlisted"] == 200
        assert (
            sum(n for c, n in counts.items() if c not in ("A-D", "unique-unlisted"))
            == 600
        )
        assert [policy["strata"][f"train/{kind}/{d}"] for d in range(1, 9)] == [
            112,
            126,
            140,
            154,
            182,
            196,
            224,
            266,
        ]
    assert sum(s.contingent for s in planned) == 24
    assert all(
        s.depth is None and s.kind == "which" and s.category == "A-D"
        for s in planned
        if s.contingent
    )
    assert all(
        s.depth is None
        for s in planned
        if s.category in ("no-entailed-answer", "inconsistent")
    )
    assert len(policy["count_targets"]) == 40
    assert sum(policy["count_targets"].values()) == 1400
    assert any(
        r["match"].get("sole_entailed_with_contingent")
        for r in policy["coverage_minima"]
    )
    validate_schedule(planned, policy)
    with pytest.raises(ValueError):
        validate_schedule(planned[:-1], policy)


def runner():
    spec = importlib.util.spec_from_file_location(
        "replacement_builder_test", EXP / "build_controlled_training.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXP))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.mark.parametrize("existing", ("output", "receipt", "manifest"))
def test_existing_output_blocks_generation(tmp_path, monkeypatch, existing):
    module = runner()
    receipt, output = tmp_path / "receipt.json", tmp_path / "candidate"
    target = {
        "output": output,
        "receipt": receipt,
        "manifest": receipt.with_suffix(".manifest.json"),
    }[existing]
    target.write_text("preserve")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--receipt",
            str(receipt),
            "--output",
            str(output),
            "--tokenizer",
            "/tmp/missing",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert target.read_text() == "preserve"
    assert len(list(tmp_path.iterdir())) == 1


def test_preflight_reports_shortage_instead_of_counting_duplicate_questions(monkeypatch):
    module = runner()
    story, problem = module.build_scene(
        module.SceneRequest(1, module.Direction.NORTH, "direction", "sole", size="compact"),
        identity="cap-fixture", family="fixture",
    )
    calls = []

    def repeated(slot, alternative):
        calls.append((slot.index, alternative))
        return story, problem, "fixture-anchor"

    monkeypatch.setattr(module, "construct", repeated)
    planned = [
        module.Slot(0, "direction", "A-D", 1, 1),
        module.Slot(1, "direction", "A-D", 1, 1),
    ]
    selected, report = module.preflight(
        planned, {"query_keys": [], "scene_keys": [], "anchor_keys": []}
    )
    assert len(selected) == 1
    assert len(calls) == report["counts"]["attempts"] == 6
    assert report["shortages"][0]["required"] == 2
    assert report["shortages"][0]["found"] == 1
