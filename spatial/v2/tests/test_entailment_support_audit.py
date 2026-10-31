"""Training-only audit selection preserves original RNG prefixes and source IDs."""

from copy import deepcopy

from spatial.v2 import composition_production as sources
from spatial.v2.entailment_support_audit import audit_schedule, summary


def policy():
    return {
        "seed": 20261030,
        "source_batches": [
            {
                "generator": "sampled-forest",
                "seed": 3,
                "seeds": [3, 7],
                "count": 2,
                "indices": [1],
                "split": "train",
                "query_kinds": ["direction", "which"],
                "depth": 2,
                "proof_pattern": "homogeneous",
                "matches": 1,
            },
            {
                "generator": "random-world",
                "seed": 1,
                "count": 2,
                "indices": [1],
                "split": "train",
                "query_kinds": ["direction", "which", "count"],
            },
            {
                "generator": "sampled-forest",
                "seed": 3,
                "count": 1,
                "split": "dev",
                "query_kinds": ["direction"],
                "depth": 2,
                "proof_pattern": "homogeneous",
                "matches": 1,
            },
        ],
    }


def test_prefix_keeps_original_problem_and_identity_without_heldout():
    spec = policy()
    original = deepcopy(spec)
    full = {
        entry.story.story_id: entry
        for entry in sources.schedule(spec)
        if entry.split == "train"
    }
    selected = audit_schedule(spec, semantic_scenes=1, targeted_scenes=1)
    assert spec == original
    assert len(selected) == 2
    assert all(entry.split == "train" for entry in selected)
    for entry in selected:
        assert entry == full[entry.story.story_id]


def test_controlled_cell_prefix_does_not_repeat_same_cell():
    spec = policy()
    extra = deepcopy(spec["source_batches"][0])
    extra["indices"] = [0]
    spec["source_batches"].append(extra)
    selected = audit_schedule(spec, semantic_scenes=1)
    assert len(selected) == 2


def test_summary_reports_failures_without_declaring_release():
    report = summary(
        [
            {
                "status": "unsupported",
                "source": "random-world",
                "base_id": "fixture",
                "reason": "missing supported proof",
            }
        ]
    )
    assert report["accepted_questions"] == 0
    assert len(report["failures"]) == 1
    assert report["production_generation"] is False
    assert report["held_out_questions_inspected"] == 0
