"""Public-interface regression checks against the preserved coverage recipes."""

from dataclasses import replace

import pytest

from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_controlled_coverage import construct, slots
from spatial.v2.entailment_controlled_prototype import check_semantics
from spatial.v2.entailment_dataset import (
    measure_depth,
    menus_for,
    selected_explanation_depth,
)
from spatial.v2.entailment_evidence import build_evidence
from spatial.v2.entailment_generator import SceneRequest, build_scene
from spatial.v2.solver import Direction


def request_for(slot):
    return SceneRequest(
        span=slot.span,
        direction=slot.direction,
        kind=slot.kind,
        case="sole" if slot.case in ("listed", "unlisted") else slot.case,
        count_value=slot.count_value,
        variant=slot.variant,
        size=("compact-extra" if slot.extra_size else "compact")
        if slot.compact
        else "contextual",
    )


@pytest.mark.parametrize("slot", slots(), ids=lambda s: s.identifier)
def test_exact_equivalence_to_every_fixed_coverage_scene(slot):
    old_story, old_problem = construct(slot)
    story, problem = build_scene(
        request_for(slot), identity=slot.identifier, family=slot.family
    )
    assert problem == old_problem
    # Only provenance changes, not names, premise order, edges or query scope.
    assert replace(story, recipe=old_story.recipe) == old_story


@pytest.mark.parametrize(
    "slot",
    [
        s
        for s in slots()
        if s.compact
        and s.extra_size == 0
        or not s.compact
        and s.span in (2, 10)
        and (
            s.case in ("no-entailed", "inconsistent")
            or s.direction in (Direction.NORTH, Direction.NORTHEAST)
        )
    ],
    ids=lambda s: s.identifier,
)
def test_public_interface_semantics_and_actual_depth(slot):
    request = request_for(slot)
    _, problem = build_scene(request, identity=slot.identifier, family=slot.family)
    solver = QuestionSolver()
    evidence = build_evidence(problem, solver=solver)
    check_semantics(slot, evidence.semantics)
    for menu in menus_for(evidence.semantics, slot.seed, omission_percent=100):
        assert (
            selected_explanation_depth(evidence, menu, Contract.UNIQUE)
            == request.requested_depth
        )
    if request.requested_depth is not None:
        assert measure_depth(problem, solver=solver)["depth"] == request.requested_depth


@pytest.mark.parametrize(
    "changes",
    [
        {"span": True},
        {"span": 0},
        {"span": 11},
        {"direction": "North"},
        {"count_value": True},
        {"count_value": 5},
        {"variant": 3},
        {"case": "listed"},
        {"case": "unlisted"},
        {"case": "multiple"},
        {"kind": "other"},
        {"size": "other"},
        {"span": 2, "size": "compact"},
    ],
)
def test_invalid_requests_fail_before_construction(changes):
    with pytest.raises(ValueError):
        SceneRequest(
            **{
                "span": 1,
                "direction": Direction.NORTH,
                "kind": "direction",
                "case": "sole",
                **changes,
            }
        )


def test_no_calls_to_historical_construction_helpers(monkeypatch):
    import spatial.v2.composition_scene_sampler as sampler
    import spatial.v2.entailment_controlled_coverage as coverage
    import spatial.v2.entailment_controlled_prototype as prototype

    request = SceneRequest(2, Direction.NORTH, "direction", "sole")
    expected = build_scene(request, identity="interface-fixture", family="fixture")

    def fail(*_args, **_kwargs):
        pytest.fail("standalone generator called a historical construction helper")

    monkeypatch.setattr(sampler, "_support", fail)
    monkeypatch.setattr(prototype, "_core", fail)
    monkeypatch.setattr(coverage, "original_core", fail)
    monkeypatch.setattr(coverage, "construct", fail)
    assert (
        build_scene(request, identity="interface-fixture", family="fixture") == expected
    )


def test_identity_validation_and_no_implicit_menu_or_label():
    request = SceneRequest(1, Direction.NORTH, "direction", "sole")
    with pytest.raises(ValueError):
        build_scene(request, identity="", family="fixture")
    with pytest.raises(ValueError):
        build_scene(request, identity="fixture", family=" ")
    with pytest.raises(TypeError):
        build_scene(None, identity="fixture", family="fixture")
    assert request.requested_depth == 1
    assert replace(request, case="no-entailed").requested_depth is None
    assert replace(request, case="inconsistent").requested_depth is None


@pytest.mark.parametrize("direction", tuple(Direction))
@pytest.mark.parametrize("variant", (0, 1))
@pytest.mark.parametrize("span", (1, 4, 10))
@pytest.mark.parametrize(
    "kind,case,count,contingent",
    [
        ("direction", "sole", 1, False),
        ("which", "sole", 1, False),
        ("which", "sole", 1, True),
        ("which", "multiple", 1, False),
        *(("count", "sole", value, False) for value in range(5)),
        *(
            (kind, case, 1, False)
            for kind in ("direction", "which", "count")
            for case in ("no-entailed", "inconsistent")
        ),
    ],
)
def test_varied_context_preserves_semantics_and_support(
    direction, variant, span, kind, case, count, contingent
):
    request = SceneRequest(span, direction, kind, case, count, variant)
    kwargs = {
        "identity": "varied-context-regression",
        "family": "test-family",
        "structure_seed": 431 + variant,
        "contingent": contingent,
    }
    _, old = build_scene(request, **kwargs)
    _, new = build_scene(request, **kwargs, varied_context=True)
    assert 0 <= len(new.objects) - len(old.objects) <= 2
    added = len(new.objects) - len(old.objects)
    assert added <= len(new.premise.operands) - len(old.premise.operands) <= added + 4
    solver = QuestionSolver()
    before = build_evidence(old, solver=solver).semantics
    evidence = build_evidence(new, solver=solver)
    after = evidence.semantics
    assert before.consistent == after.consistent == (case != "inconsistent")
    # New objects can change the name mapping, so compare semantic obligations.
    if case == "sole":
        assert len(after.entailed) == 1
        if kind == "count":
            assert after.entailed == {count}
    elif case == "multiple":
        assert len(before.entailed) == len(after.entailed) >= 2
    else:
        assert not after.entailed
    if contingent:
        assert "contingent" in solver.negative_categories(new).values()
    if case not in ("no-entailed", "inconsistent"):
        if not contingent:
            assert measure_depth(new, solver=solver)["depth"] == span
        for menu in menus_for(after, 17, omission_percent=100):
            assert selected_explanation_depth(evidence, menu, Contract.UNIQUE) == span


@pytest.mark.parametrize(
    "size,seed,varied",
    [
        ("contextual", 42, 1),
        ("compact", 42, True),
        ("contextual", None, True),
    ],
)
def test_invalid_varied_context_requests_are_rejected(size, seed, varied):
    with pytest.raises(ValueError):
        build_scene(
            SceneRequest(1, Direction.NORTH, "direction", "sole", size=size),
            identity="invalid-crosslinks",
            family="fixture",
            structure_seed=seed,
            varied_context=varied,
        )


@pytest.mark.parametrize("span", range(1, 11))
def test_context_size_varies_at_every_level_with_same_case_size_law(span):
    sizes = set()
    for seed in range(12):
        pair_sizes = []
        for case in ("sole", "no-entailed", "inconsistent"):
            _, problem = build_scene(
                SceneRequest(span, Direction.NORTH, "count", case),
                identity="context-size",
                family="fixture",
                structure_seed=seed,
                varied_context=True,
            )
            pair_sizes.append((len(problem.objects), len(problem.premise.operands)))
        assert len(set(pair_sizes)) == 1
        sizes.add(pair_sizes[0])
    assert len(sizes) > 1


@pytest.mark.parametrize(
    "varied,expected",
    [
        (False, "1e7d9080482e5308e4c4f4aeee54ab444ef3686ceb58f114da370573d41b4ab2"),
        (True, "cf1df9a71453ac827d60340dcdb2351345ac8f9b3c360ba95e1b22792b74d022"),
    ],
)
def test_seeded_output_matches_precleanup_baseline(varied, expected):
    import hashlib
    import json

    from spatial.v2.serialization import tagged_dataclass_to_dict

    cases = [
        ("direction", "sole", 1, False),
        ("which", "sole", 1, False),
        ("which", "sole", 1, True),
        ("which", "multiple", 1, False),
        *(("count", "sole", v, False) for v in range(5)),
        *(
            (k, c, 1, False)
            for k in ("direction", "which", "count")
            for c in ("no-entailed", "inconsistent")
        ),
    ]
    digest = hashlib.sha256()
    for span in range(1, 11):
        for direction in Direction:
            for kind, case, value, contingent in cases:
                for variant in (0, 1):
                    for seed in (0, 431):
                        story, problem = build_scene(
                            SceneRequest(span, direction, kind, case, value, variant),
                            identity="cleanup-output-lock",
                            family="fixture",
                            structure_seed=seed,
                            contingent=contingent,
                            varied_context=varied,
                        )
                        digest.update(
                            json.dumps(
                                [
                                    tagged_dataclass_to_dict(story),
                                    tagged_dataclass_to_dict(problem),
                                ],
                                sort_keys=True,
                            ).encode()
                        )
    assert digest.hexdigest() == expected


@pytest.mark.parametrize("span", range(1, 11))
@pytest.mark.parametrize("direction", tuple(Direction))
@pytest.mark.parametrize("variant", (0, 1))
@pytest.mark.parametrize("upper", (False, True))
@pytest.mark.parametrize(
    "kind,case,count,contingent",
    [
        ("direction", "sole", 1, False),
        ("which", "sole", 1, False),
        ("which", "sole", 1, True),
        ("which", "multiple", 1, False),
        *(("count", "sole", v, False) for v in range(5)),
        *(
            (k, c, 1, False)
            for k in ("direction", "which", "count")
            for c in ("no-entailed", "inconsistent")
        ),
    ],
)
def test_explicit_endpoints_preserve_semantics_and_depth(
    span, direction, variant, upper, kind, case, count, contingent
):
    pair = (2 * span + 8, 2 * span + 14) if upper else (2 * span + 4, 2 * span + 6)
    request = SceneRequest(span, direction, kind, case, count, variant)
    _, problem = build_scene(
        request,
        identity="explicit-endpoint",
        family="fixture",
        structure_seed=42,
        contingent=contingent,
        size_pair=pair,
    )
    assert (len(problem.objects), len(problem.premise.operands)) == pair
    solver = QuestionSolver()
    evidence = build_evidence(problem, solver=solver)
    assert evidence.semantics.consistent == (case != "inconsistent")
    if case == "sole":
        assert len(evidence.semantics.entailed) == 1
        if kind == "count":
            assert evidence.semantics.entailed == {count}
    elif case == "multiple":
        assert len(evidence.semantics.entailed) == 2
    else:
        assert not evidence.semantics.entailed
    if contingent:
        assert "contingent" in solver.negative_categories(problem).values()
    if case in ("sole", "multiple"):
        for menu in menus_for(evidence.semantics, 42, omission_percent=100):
            assert selected_explanation_depth(evidence, menu, Contract.UNIQUE) == span
        if not contingent:
            assert measure_depth(problem, solver=solver)["depth"] == span


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size_pair": (True, 8)},
        {"size_pair": (6,)},
        {"size_pair": (32, 34)},
        {"size_pair": (6, 16)},
        {"size_pair": (6, 8), "varied_context": True},
        {"size_pair": (6, 8), "structure_seed": None},
        {"size_pair": (6, 8), "augmentation_limit": -1},
        {"size_pair": (6, 8), "augmentation_limit": True},
    ],
)
def test_invalid_explicit_sizes_fail_without_clamping(kwargs):
    with pytest.raises(ValueError, match="explicit-size"):
        build_scene(
            SceneRequest(1, Direction.NORTH, "direction", "sole"),
            identity="invalid-size",
            family="fixture",
            **{"structure_seed": 42, **kwargs},
        )


def test_explicit_size_smaller_than_core_is_not_increased():
    with pytest.raises(ValueError, match="smaller than semantic core"):
        build_scene(
            SceneRequest(10, Direction.NORTH, "count", "sole", 4),
            identity="small-core",
            family="fixture",
            structure_seed=42,
            size_pair=(6, 8),
        )


def test_explicit_construction_propagates_inconclusive_solver(monkeypatch):
    from types import SimpleNamespace

    from spatial.v2.composition_contracts import VerificationError

    monkeypatch.setattr(
        QuestionSolver,
        "assess",
        lambda *_args, **_kwargs: SimpleNamespace(error="test timeout"),
    )
    with pytest.raises(VerificationError, match="inconclusive contingent analysis"):
        build_scene(
            SceneRequest(1, Direction.NORTH, "which", "sole"),
            identity="inconclusive",
            family="fixture",
            structure_seed=42,
            contingent=True,
            size_pair=(6, 8),
        )
