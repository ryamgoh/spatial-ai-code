import json
from collections import Counter

import pytest

from spatial.v2.generation import QueryKind
from spatial.v2.study.randomized_v2 import (
    RandomizedStudyDesign,
    _calibration_row,
    _duplicate_structure_indices,
    benchmark_partition_quotas,
    calibration_length_quotas,
    pair_marginal_quotas,
    prepare_calibration,
    prepare_headline_benchmark,
    study_manifest,
    training_assignments,
    write_or_verify_manifest,
    write_consistency_diagnostic,
    write_special_response_diagnostic,
)
from spatial.v2.study import randomized_v2


def test_every_query_gold_group_keeps_three_hundred_headline_examples() -> None:
    for query in QueryKind:
        for gold in range(1, 5):
            quotas = benchmark_partition_quotas(query, gold)
            assert sum(quota.count for quota in quotas) == 300
            for quota in quotas:
                assert sum(quota.length.values()) == quota.count
                assert sum(quota.distractors.values()) == quota.count


def test_count_gold_dependent_minimum_is_reflected_without_shrinking_group() -> None:
    quotas = benchmark_partition_quotas(QueryKind.COUNT, 4)
    matched = quotas[0]

    assert set(matched.length) == {4, 5, 6}
    assert sum(matched.length.values()) == 120
    assert calibration_length_quotas(QueryKind.COUNT, 4, total=10) == {
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
    }


@pytest.mark.parametrize(
    ("gold", "expected_lengths"),
    (
        (1, {2, 3, 4, 5, 6}),
        (2, {4, 5, 6}),
        (3, {5, 6}),
        (4, {6}),
    ),
)
def test_count_composition_reserves_two_prefix_steps_for_heldout_edge(
    gold: int, expected_lengths: set[int]
) -> None:
    composition = next(
        quota
        for quota in benchmark_partition_quotas(QueryKind.COUNT, gold)
        if quota.partition is randomized_v2.BenchmarkPartition.COMPOSITION_OOD
    )

    assert set(composition.length) == expected_lengths
    assert sum(composition.length.values()) == 60


def test_study_manifest_has_expected_contract_totals() -> None:
    manifest = study_manifest()

    assert len(manifest["benchmark"]) == 12
    assert manifest["contracts"]["single"]["headline_examples"] == 900
    assert manifest["contracts"]["all-possible"]["headline_examples"] == 2700
    assert sum(manifest["training"]["single_length"].values()) == 8000
    assert all(
        sum(quotas.values()) == 2000
        for quotas in manifest["training"]["multi_length_by_gold"].values()
    )
    assert sum(manifest["training"]["single_distractors"].values()) == 8000
    assert all(
        sum(quotas.values()) == 2000
        for quotas in manifest["training"]["multi_distractors_by_gold"].values()
    )


def test_study_manifest_scopes_answer_only_ablation_to_single() -> None:
    training = study_manifest()["training"]

    assert training["examples_per_contract"] == 8000
    assert training["development_per_contract"] == 800
    assert training["supervision_by_contract"] == {
        "single": ["natural", "answer-only"],
        "multi": ["natural"],
    }
    assert "examples_per_arm" not in training
    assert "development_per_arm" not in training
    assert "supervision" not in training


def test_identical_cancelled_run_manifest_is_reused_but_drift_is_rejected(
    tmp_path,
) -> None:
    path = tmp_path / "study-manifest.json"
    manifest = study_manifest()

    assert write_or_verify_manifest(path, manifest) == "written"
    assert write_or_verify_manifest(path, manifest) == "reused"
    with pytest.raises(FileExistsError, match="different study design"):
        write_or_verify_manifest(
            path,
            study_manifest(RandomizedStudyDesign(train_max_x=5)),
        )


def test_benchmark_rejects_calibration_with_superseded_study_manifest(
    tmp_path,
) -> None:
    calibration = {
        "schema": "spatialentail-randomized-v2-calibration-v1",
        "rows": 2400,
        "transition_probe": {},
        "study": study_manifest(),
    }
    calibration["study"]["benchmark"]["count-g2"][-1]["length"] = {"2": 60}
    manifest_path = tmp_path / "calibration.json"
    manifest_path.write_text(json.dumps(calibration))

    with pytest.raises(ValueError, match="superseded study manifest"):
        prepare_headline_benchmark(
            tmp_path / "benchmark",
            calibration_manifest=manifest_path,
            workers=0,
        )


def test_calibration_duplicate_detection_reports_rows_without_rejecting_pool() -> None:
    rows = [
        {"metadata": {"structure_signature": "a"}},
        {"metadata": {"structure_signature": "b"}},
        {"metadata": {"structure_signature": "a"}},
        {"metadata": {"structure_signature": "b"}},
    ]

    assert _duplicate_structure_indices(rows) == [2, 3]


def test_transition_probe_threshold_scales_with_attempt_count() -> None:
    from spatial.v2.study.randomized_v2 import _capability_success_threshold

    assert _capability_success_threshold(20) == 16
    assert _capability_success_threshold(10) == 8


def test_marginal_pairing_preserves_both_exact_distributions() -> None:
    pairs = pair_marginal_quotas({1: 2, 2: 3}, {0: 1, 1: 4}, seed=42)

    assert len(pairs) == 5
    assert [item[0] for item in pairs].count(1) == 2
    assert [item[0] for item in pairs].count(2) == 3
    assert [item[1] for item in pairs].count(0) == 1
    assert [item[1] for item in pairs].count(1) == 4


def test_marginal_pairing_respects_predeclared_joint_capacity() -> None:
    lengths = {4: 34, 5: 40, 6: 46}
    distractors = {0: 13, 1: 14, 2: 15, 3: 17, 4: 19, 5: 20, 6: 22}
    pairs = pair_marginal_quotas(
        lengths,
        distractors,
        seed=42,
        pair_caps={(4, 0): 4},
    )

    assert Counter(length for length, _ in pairs) == lengths
    assert Counter(distractor for _, distractor in pairs) == distractors
    assert pairs.count((4, 0)) <= 4


def test_minimum_count_cell_declares_empirical_operational_cap() -> None:
    matched = benchmark_partition_quotas(QueryKind.COUNT, 4)[0]

    assert matched.pair_caps == ((4, 0, 4),)
    assert matched.pair_cap_basis == "empirical-operational"


def test_multi_training_assignment_preserves_marginals_and_gold_minima() -> None:
    assignments = training_assignments("multi", total=8000, seed=42)

    assert len(assignments) == 8000
    assert {gold: sum(item[0] == gold for item in assignments) for gold in range(1, 5)} == {
        1: 2000,
        2: 2000,
        3: 2000,
        4: 2000,
    }
    counts_by_length = {
        length: sum(item[1] == length for item in assignments)
        for length in range(1, 7)
    }
    assert counts_by_length == {
        1: 122,
        2: 355,
        3: 778,
        4: 1547,
        5: 2166,
        6: 3032,
    }
    assert all(length >= gold for gold, length, _ in assignments)
    for gold in range(1, 5):
        assert {
            distractor: sum(
                item[0] == gold and item[2] == distractor for item in assignments
            )
            for distractor in range(0, 7)
        } == {0: 133, 1: 166, 2: 207, 3: 259, 4: 324, 5: 405, 6: 506}


def test_default_train_l6_x6_boundary_drives_assignments_and_benchmark() -> None:
    design = RandomizedStudyDesign()
    assert (design.train_max_l, design.train_max_x) == (6, 6)
    assert (design.eval_max_l, design.eval_max_x) == (8, 8)

    assignments = training_assignments(
        "multi", total=8000, seed=42, design=design
    )

    assert max(length for _, length, _ in assignments) == 6
    assert max(distractor for _, _, distractor in assignments) == 6
    assert all(length >= gold for gold, length, _ in assignments)
    quotas = benchmark_partition_quotas(QueryKind.COUNT, 4, design)
    assert set(quotas[0].length) == {4, 5, 6}
    assert set(quotas[1].length) == {7, 8}
    assert set(quotas[2].distractors) == {7, 8}


def test_randomized_policy_uses_bounded_object_domains() -> None:
    single = randomized_v2._randomized_policy(QueryKind.DIRECTION, 1)
    possible = randomized_v2._randomized_policy(QueryKind.COUNT, 4)

    assert single.num_entities == possible.num_entities
    assert single.num_entities < 30
    assert possible.possibility_candidate_count == 5
    assert possible.ambiguity_size == 4


@pytest.mark.parametrize(
    ("query", "gold", "seed"),
    (
        ("direction", 1, 801),
        ("direction", 4, 802),
        ("which", 4, 803),
        ("count", 4, 804),
    ),
)
def test_l8_x8_stress_cases_replay_exact_requested_complexity(
    query: str, gold: int, seed: int
) -> None:
    row = _calibration_row(query, gold, 8, 8, seed)
    metadata = row["metadata"]

    assert metadata["difficulty"]["proof_complexity"]["semantic_length"] == 8
    assert metadata["difficulty"]["num_distractor_premises"] == 8
    assert metadata["menu_option_count"] == 5
    assert len(metadata["possible_values"]) == gold
    assert len(metadata["oracle_letters"]) == gold
    assert "CSR" not in metadata["difficulty"]["requested_rule_path"]


def test_consistency_diagnostic_writes_all_clean_dirty_pairs(tmp_path) -> None:
    receipt = write_consistency_diagnostic(
        tmp_path / "diagnostic", seed=42, pairs_per_cycle_axis=1
    )

    assert receipt["pairs"] == 14
    assert receipt["rows"] == 28
    assert len((tmp_path / "diagnostic/consistency.jsonl").read_text().splitlines()) == 28


def test_special_response_smoke_covers_each_semantic_category(tmp_path) -> None:
    receipt = write_special_response_diagnostic(
        tmp_path / "special",
        seed=42,
        l0_per_query=1,
        ambiguous_per_query=1,
        no_match_count=1,
        visible_per_query_gold=1,
    )

    assert receipt["rows"] == 16
    assert receipt["categories"] == {
        "direct-l0": 3,
        "single-ambiguous": 3,
        "selection-no-match": 1,
        "visible-zero": 9,
    }


def test_calibration_smoke_uses_standard_generation_pipeline(tmp_path) -> None:
    receipt = prepare_calibration(
        tmp_path / "calibration",
        seed=42,
        examples_per_query_gold=1,
        workers=1,
        query_kinds=(QueryKind.DIRECTION,),
        gold_sizes=(1,),
        run_transition_probe=False,
    )

    assert receipt["rows"] == 1
    assert len(
        (tmp_path / "calibration/source-natural.jsonl").read_text().splitlines()
    ) == 1
