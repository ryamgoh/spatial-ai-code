"""Shared proofs must preserve exact directions, contracts and audit evidence."""

from copy import deepcopy
from dataclasses import replace

import pytest

from spatial.v2.composition_audit import aligned_pair
from spatial.v2.composition_compact import exclusion_groups, render_pair
from spatial.v2.composition_contracts import (
    VerificationError,
    encode,
    parse_answer,
    render_prompt,
)
from spatial.v2.composition_evidence import build_evidence
from spatial.v2.composition_evidence import render_pair as legacy_pair
from spatial.v2.composition_rerender import rerender_record
from spatial.v2.grading import AnswerMode
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import (
    And,
    Direction,
    DirectionQuery,
    SpatialProblem,
    SpatialSolverV2,
    direction_constraint,
)
from spatial.v2.tests.test_pivot_evidence import BY_ID, VIEWS


def chain(direction, depth=10):
    objects = tuple(f"O{i}" for i in range(depth + 1))
    return SpatialProblem(
        objects,
        And(
            tuple(
                direction_constraint(objects[i + 1], objects[i], direction)
                for i in range(depth)
            )
        ),
        DirectionQuery(objects[-1], objects[0]),
    )


@pytest.mark.parametrize("direction", tuple(Direction))
def test_exclusion_partition_matches_independent_exact_sign_table(direction):
    signs = {
        "North": (0, 1),
        "Northeast": (1, 1),
        "East": (1, 0),
        "Southeast": (1, -1),
        "South": (0, -1),
        "Southwest": (-1, -1),
        "West": (-1, 0),
        "Northwest": (-1, 1),
    }
    groups = exclusion_groups(direction)
    excluded = [d for values in groups.values() for d in values]
    assert len(excluded) == len(set(excluded)) == 7
    assert set(excluded) == set(signs) - {direction.value}
    for i, axis in enumerate(("x", "y")):
        assert all(signs[d][i] != signs[direction.value][i] for d in groups[axis])


@pytest.mark.parametrize("direction", tuple(Direction))
@pytest.mark.parametrize("listed", (True, False))
def test_all_contracts_reuse_one_deep_proof_without_mutating_evidence(
    direction, listed
):
    evidence = build_evidence(chain(direction))
    original = deepcopy(tagged_dataclass_to_dict(evidence))
    others = [d.value for d in Direction if d != direction]
    menu = tuple([direction.value, *others[:3]] if listed else others[:4])
    for mode in AnswerMode:
        pair = render_pair(evidence, menu, mode)
        expected = encode(evidence.semantics, menu, mode).letters
        assert all(parse_answer(t, mode, stage=1) == expected for t in pair.values())
        aligned_pair({"id": "test", "references": pair})
        if mode is AnswerMode.ALL_POSSIBLE and not listed:
            assert pair == legacy_pair(
                evidence, menu, mode, compact_unique_direction=True
            )
        else:
            for text in pair.values():
                assert text.count("Q-DIR:") == 1
                assert "A-REFUTE:" not in text
                assert text.count("D-X-9:") == 1
                assert text.count("D-Y-9:") == 1
                assert "The complete possible set" in text
        assert tagged_dataclass_to_dict(evidence) == original


@pytest.mark.parametrize("name,menu,expected", VIEWS)
def test_fixtures_and_unchanged_nonunique_or_nondirection_paths(name, menu, expected):
    problem = BY_ID[name].problem()
    evidence = build_evidence(problem)
    modes = (AnswerMode.SINGLE, AnswerMode.VISIBLE_POSSIBLE, AnswerMode.ALL_POSSIBLE)
    for mode, answer in zip(modes, expected):
        pair = render_pair(evidence, menu, mode)
        assert all(parse_answer(t, mode, stage=1) == answer for t in pair.values())
        aligned_pair({"id": name, "references": pair})
        if (
            not isinstance(problem.query, DirectionQuery)
            or evidence.semantics.case != "unique"
        ):
            assert pair == legacy_pair(
                evidence, menu, mode, compact_unique_direction=True
            )


def test_corrupted_full_audit_evidence_is_rejected_not_hidden_by_compaction():
    evidence = build_evidence(chain(Direction.NORTH, 2))
    changed = replace(
        evidence, semantics=replace(evidence.semantics, possible=frozenset({"South"}))
    )
    with pytest.raises(VerificationError):
        render_pair(
            changed, ("North", "East", "South", "West"), AnswerMode.ALL_POSSIBLE
        )


def test_replay_preserves_source_rows_and_enforces_whole_pair_token_admission(
    monkeypatch,
):
    problem = chain(Direction.NORTH, 2)
    evidence = build_evidence(problem)
    menu = ("North", "East", "South", "West")
    mode = AnswerMode.ALL_POSSIBLE
    row = {
        "id": "test",
        "prompt": render_prompt(problem, menu, mode),
        "menu": list(menu),
        "contract": mode.value,
        "semantic_case": "unique",
        "answer": "A",
        "references": legacy_pair(evidence, menu, mode),
    }
    original = {
        "status": "accepted",
        "problem": tagged_dataclass_to_dict(problem),
        "data": {"rows": [row], "evidence": tagged_dataclass_to_dict(evidence)},
    }
    snapshot = deepcopy(original)
    monkeypatch.setattr(
        "spatial.v2.composition_rerender.token_measurement",
        lambda tokenizer, row, style: {"admitted": style == "symbolic"},
    )
    result = rerender_record(original, None, SpatialSolverV2(backend="z3"))
    assert original == snapshot
    assert result["status"] == "token-rejected"
    for key in ("prompt", "menu", "contract", "answer", "semantic_case"):
        assert result["data"]["rows"][0][key] == row[key]
    assert result["data"]["rows"][0]["references"] != row["references"]
    original["data"]["rows"][0]["answer"] = "B"
    with pytest.raises(ValueError, match="gold"):
        rerender_record(original, None, SpatialSolverV2(backend="z3"))


def test_replay_retains_unsupported_attempts():
    original = {"status": "unsupported", "reason": "example"}
    assert rerender_record(original, None, None) == original
