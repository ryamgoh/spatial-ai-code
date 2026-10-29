from copy import deepcopy

import pytest

from spatial.v2.composition_selection import (
    features,
    inventory,
    select,
    validate_selection,
)


def bundles():
    result = []
    for kind in ("direction", "which", "count"):
        for i in range(3):
            result.append(
                {
                    "id": f"{kind}-{i}",
                    "base": f"{kind}-{i}",
                    "kind": kind,
                    "stratum": (kind, "shared-topology", i),
                    "topology": "shared-topology",
                    "features": {(kind, "depth", 1 if i == 0 else 2)},
                    "cells": {f"{kind}/{i}"},
                    "escapes": int(i == 2),
                }
            )
    return result


def test_coverage_survives_shared_topology():
    pool = bundles()
    result = select(pool, quota=2)
    assert result["status"] == "optimal"
    assert set(result["bundle_ids"]) == {
        f"{kind}-{i}" for kind in ("direction", "which", "count") for i in (0, 1)
    }
    validate_selection(pool, result["bundle_ids"], 2)


def test_infeasible_coverage_does_not_relax():
    assert select(bundles(), quota=1)["status"] == "unsat"


def test_duplicate_base_not_used_to_fill_quota():
    pool = bundles()
    for b in pool:
        b["base"] = b["kind"]
    assert select(pool, quota=2)["status"] == "unsat"


def test_same_stratum_topology_cap_enforced():
    pool = bundles()
    for b in pool:
        b["stratum"] = (b["kind"],)
    assert select(pool, quota=2)["status"] == "unsat"


def test_validator_rejects_coverage_loss_and_duplicates():
    pool = bundles()
    ids = [f"{k}-{i}" for k in ("direction", "which", "count") for i in (1, 2)]
    with pytest.raises(ValueError, match="coverage"):
        validate_selection(pool, ids, 2)
    with pytest.raises(ValueError, match="repeated"):
        validate_selection(pool, ids + ids[:1], 2)


def test_no_mutation_and_stable_selection():
    pool = bundles()
    original = deepcopy(pool)
    a = select(pool, quota=2)
    assert a == select(pool, quota=2)
    assert pool == original


def test_escape_cannot_supply_depth():
    row = {
        "query_kind": "which",
        "contract": "single",
        "semantic_case": "no-match",
        "returned_count": None,
        "escape_reason": "no-match",
        "substantive_depth_support": False,
        "depth": 10,
    }
    assert not any(f[0] == "depth" for f in features(row))
    row["substantive_depth_support"] = True
    with pytest.raises(ValueError):
        features(row)


def test_inventory_accepts_contract_specific_prompts():
    from spatial.v2.composition_contracts import render_prompt
    from spatial.v2.grading import AnswerMode
    from spatial.v2.tests.test_pivot_evidence import BY_ID

    menu = ("West", "Northeast", "South", "Northwest")
    rows = [
        {
            "base_id": "base",
            "bundle_id": "menu",
            "query_kind": "direction",
            "contract": mode.value,
            "menu": list(menu),
            "prompt": render_prompt(BY_ID["D1"].problem(), menu, mode),
            "references": {"natural": "Answer: B", "symbolic": "Answer: B"},
            "answer": "B",
            "semantic_case": "unique",
            "returned_count": 1,
            "escape_reason": None,
            "possible_count": 1,
            "displayed_possible_count": 1,
            "contains_zero": False,
            "depth": 2,
            "substantive_depth_support": True,
        }
        for mode in AnswerMode
    ]
    record = {
        "status": "accepted",
        "base_id": "base",
        "query_kind": "direction",
        "family_id": "family",
        "recipe": "test",
        "structure": {"core_topology": "topology"},
        "rows": rows,
    }
    assert len(inventory([record])) == 1
    rows[0]["menu"] = list(reversed(menu))
    with pytest.raises(ValueError, match="mismatched"):
        inventory([record])


def test_milp_matches_z3_on_small_instance():
    from spatial.v2.composition_selection import select_milp

    pool = bundles()
    a = select(pool, quota=2)
    b = select_milp(pool, quota=2)
    assert b["status"] == "optimal"
    assert b["bundle_ids"] == a["bundle_ids"]
    assert select_milp(pool, quota=1)["status"] == "unsat"


def test_milp_preserves_exact_cells_before_reducing_escapes():
    from spatial.v2.composition_selection import select_milp

    pool = bundles()
    required = {"which/2"}
    result = select_milp(pool, quota=2, required_cells=required)
    assert result["status"] == "optimal"
    assert "which-2" in result["bundle_ids"]
    validate_selection(pool, result["bundle_ids"], 2, required_cells=required)
    original = select_milp(pool, quota=2)["bundle_ids"]
    with pytest.raises(ValueError, match="fine-cell"):
        validate_selection(pool, original, 2, required_cells=required)
    assert select_milp(pool, quota=2, required_cells={"absent"})["status"] == "unsat"


def test_milp_recipe_caps_are_hard_and_independently_checked():
    from spatial.v2.composition_selection import select_milp

    pool = bundles()
    for b in pool:
        b["recipe"] = "new" if b["id"].endswith("-1") else "old"
    caps = {("which", "new"): 0}
    result = select_milp(pool, quota=2, recipe_caps=caps)
    assert result["status"] == "optimal"
    assert "which-1" not in result["bundle_ids"]
    validate_selection(pool, result["bundle_ids"], 2, recipe_caps=caps)
    with pytest.raises(ValueError, match="recipe cap"):
        validate_selection(
            pool, select_milp(pool, quota=2)["bundle_ids"], 2, recipe_caps=caps
        )
    assert (
        select_milp(pool, quota=2, required_cells={"which/1"}, recipe_caps=caps)[
            "status"
        ]
        == "unsat"
    )
    with pytest.raises(ValueError, match="nonnegative integers"):
        select_milp(pool, quota=2, recipe_caps={("which", "new"): -1})
