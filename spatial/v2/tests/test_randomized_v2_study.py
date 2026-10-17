from spatial.v2.generation import QueryKind
from spatial.v2.study.randomized_v2 import (
    RandomizedStudyDesign,
    benchmark_partition_quotas,
    calibration_length_quotas,
    pair_marginal_quotas,
    prepare_calibration,
    study_manifest,
    training_assignments,
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

    assert set(matched.length) == {4, 5}
    assert sum(matched.length.values()) == 120
    assert calibration_length_quotas(QueryKind.COUNT, 4, total=10) == {
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
    }


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


def test_marginal_pairing_preserves_both_exact_distributions() -> None:
    pairs = pair_marginal_quotas({1: 2, 2: 3}, {0: 1, 1: 4}, seed=42)

    assert len(pairs) == 5
    assert [item[0] for item in pairs].count(1) == 2
    assert [item[0] for item in pairs].count(2) == 3
    assert [item[1] for item in pairs].count(0) == 1
    assert [item[1] for item in pairs].count(1) == 4


def test_multi_training_assignment_preserves_marginals_and_gold_minima() -> None:
    assignments = training_assignments("multi", total=8000, seed=42)

    assert len(assignments) == 8000
    assert {gold: sum(item[0] == gold for item in assignments) for gold in range(1, 5)} == {
        1: 2000,
        2: 2000,
        3: 2000,
        4: 2000,
    }
    assert {length: sum(item[1] == length for item in assignments) for length in range(1, 6)} == {
        1: 183,
        2: 538,
        3: 1211,
        4: 2528,
        5: 3540,
    }
    assert all(length >= gold for gold, length, _ in assignments)
    for gold in range(1, 5):
        assert {
            distractor: sum(
                item[0] == gold and item[2] == distractor for item in assignments
            )
            for distractor in range(0, 6)
        } == {0: 178, 1: 222, 2: 277, 3: 347, 4: 434, 5: 542}


def test_train_l6_x6_boundary_is_parameterized_not_a_code_branch() -> None:
    design = RandomizedStudyDesign(train_max_l=6, train_max_x=6)
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

    assert single.num_entities == 24
    assert possible.num_entities == 24
    assert possible.possibility_candidate_count == 5


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
