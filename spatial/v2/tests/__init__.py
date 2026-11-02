"""V2 spatial reasoning contract tests."""


def composition_policy(*, coverage=False):
    """Fresh test-only policy matching the retired numerical/gate fixtures.

    Values come from the pre-e340840 policies, not the current experiment.
    There are deliberately no production source batches or real approval receipts.
    """
    targets = {
        "train": 6000,
        "dev": 1200,
        "in-range-test": 2400,
        "extrapolation-test": 1200,
    }
    depths = {
        role: range(9, 11) if role == "extrapolation-test" else range(1, 9)
        for role in targets
    }
    kinds = ("direction", "which", "count")
    return {
        "schema": "composition-production-v1",
        "status": "numeric-approved",
        "approval": {"production_freeze": False, "training_authorized": False},
        "seed": 20261102,
        "base_targets": targets,
        "depth_cohort_percent": 70,
        "source_mix_percent": {
            "controlled": 70,
            "random-world": 15,
            "random-atoms": 10,
            "targeted": 5,
        },
        "depth_weights": {
            "train": dict(zip(map(str, range(1, 9)), (8, 9, 10, 11, 13, 14, 16, 19))),
            **{
                role: dict.fromkeys(map(str, depths[role]), 1)
                for role in targets
                if role != "train"
            },
        },
        "paired_omission_percent": 20,
        "controlled_count_weights": dict.fromkeys(map(str, range(5)), 1),
        "diversity_checks": {
            "status": "approved",
            "semantic_minimum_percent": 5,
            "contract_difference_semantic_percent": 20,
            "contract_difference_families": {"train": 10, "dev": 3, "in-range-test": 5},
            "nonhomogeneous_controlled_percent": 30,
        },
        "depth_family_minima": {
            f"{role}/{kind}/{depth}": floor
            for role, floor in zip(targets, (50, 10, 20, 50))
            for kind in kinds
            for depth in depths[role]
        },
        "scene_limits": {"objects": 32, "premises": 32, "candidates": 8},
        "solver_timeout_ms": 5000,
        "attempt_budget": 32400,
        "source_batches": None,
        "development_exclusions": None,
        "coverage_minima": [
            {
                "match": {
                    "split": role,
                    "query_kind": kind,
                    "contract": contract,
                    "menu_role": "primary",
                    "substantive_depth_support": True,
                },
                "minimum_bases": minimum,
            }
            for role, minimum in zip(targets, (1400, 280, 560, 400))
            for kind in kinds
            for contract in ("all-possible", "single", "visible-possible")
        ]
        if coverage
        else None,
    }
