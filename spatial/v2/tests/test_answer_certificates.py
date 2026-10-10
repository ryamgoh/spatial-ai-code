"""Contracts for complete Direction answer-set certificates."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from spatial.v2.answer_certificate_renderers import (
    render_direction_answer_set,
    render_direction_training_trace,
)
from spatial.v2.answer_certificates import (
    AnswerSetCheckError,
    DirectionAnswerSetCertificate,
    DirectionCandidateCertificate,
    DirectionEntailmentCertificate,
    answer_set_to_dict,
    build_direction_answer_set,
    check_direction_answer_set,
)
from spatial.v2.proofs import ProofConstructionError
from spatial.v2.solver import (
    And,
    Direction,
    DirectionQuery,
    RelationConstraint,
    SpatialProblem,
)
from spatial.v2.symbolic_trace_codec import parse_symbolic_direction_trace
from spatial.v2.trace import TraceFormat


def atom(subject: str, direction: Direction, reference: str) -> RelationConstraint:
    return RelationConstraint(subject, reference, frozenset({direction}))


def test_unique_answer_set_covers_all_eight_candidates() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=northeast,
        query=DirectionQuery("A", "B"),
    )

    certificate = build_direction_answer_set(
        problem,
        {Direction.NORTHEAST: {"A": (1, 1), "B": (0, 0)}},
    )

    check_direction_answer_set(certificate)
    assert certificate.possible_directions == (Direction.NORTHEAST,)
    assert certificate.is_unique
    assert certificate.entailed_directions == (Direction.NORTHEAST,)
    assert isinstance(
        next(item for item in certificate.candidates if item.possible).evidence,
        DirectionEntailmentCertificate,
    )
    natural = render_direction_training_trace(
        certificate,
        TraceFormat.NATURAL,
    )
    symbolic = render_direction_training_trace(
        certificate,
        TraceFormat.SYMBOLIC,
    )
    parsed = parse_symbolic_direction_trace(problem, symbolic)
    assert "as established by P1" in natural
    assert "supporting model" in natural
    assert len(parsed.models) == 1
    assert natural.count("P1:") == 1
    assert "all other exact directions are excluded" in natural
    assert natural.index("Q-DIR:") < natural.index("unique direction is Northeast")
    assert natural.index("Q-DIR:") < natural.index("supporting model")
    assert parsed.possible_directions == (Direction.NORTHEAST,)
    assert parsed.proof is not None
    assert parsed.proof.conclusion.direction is Direction.NORTHEAST

    audit = render_direction_answer_set(
        certificate,
        TraceFormat.NATURAL,
    )
    assert "supporting model" in audit
    assert audit.count("Assume for contradiction") == 7
    assert len(certificate.candidates) == len(Direction)
    assert sum(not item.possible for item in certificate.candidates) == 7
    json.dumps(answer_set_to_dict(certificate))


def test_ambiguous_ordinal_answer_set_has_two_models_and_two_refutations() -> None:
    objects = ("N", "R", "S")
    problem = SpatialProblem(
        objects=objects,
        premise=And(
            (
                atom("S", Direction.SOUTHEAST, "R"),
                atom("N", Direction.SOUTHWEST, "S"),
            )
        ),
        query=DirectionQuery(
            "R",
            "N",
            frozenset(
                {
                    Direction.NORTHEAST,
                    Direction.NORTHWEST,
                    Direction.SOUTHEAST,
                    Direction.SOUTHWEST,
                }
            ),
        ),
    )
    certificate = build_direction_answer_set(
        problem,
        {
            Direction.NORTHEAST: {
                "N": (0, 0),
                "R": (1, 2),
                "S": (2, 1),
            },
            Direction.NORTHWEST: {
                "N": (1, 0),
                "R": (0, 2),
                "S": (2, 1),
            },
        },
    )

    assert certificate.possible_directions == (
        Direction.NORTHEAST,
        Direction.NORTHWEST,
    )
    assert not certificate.is_unique
    assert certificate.entailed_directions == ()
    assert sum(item.possible for item in certificate.candidates) == 2
    assert sum(not item.possible for item in certificate.candidates) == 2
    natural = render_direction_training_trace(
        certificate,
        TraceFormat.NATURAL,
    )
    symbolic = render_direction_training_trace(
        certificate,
        TraceFormat.SYMBOLIC,
    )
    parsed = parse_symbolic_direction_trace(problem, symbolic)
    assert "X west to east:" in natural
    assert natural.index("X west to east:") < natural.index("Northeast: possible")
    assert parsed.possible_directions == (
        Direction.NORTHEAST,
        Direction.NORTHWEST,
    )
    assert len(parsed.models) == 2
    assert len(parsed.refutations) == 2
    assert "possible directions are Northeast, Northwest" in natural
    assert '"schema":"spatial-direction-trace-v2"' in symbolic


def test_builder_rejects_a_missing_model_for_a_possible_candidate() -> None:
    problem = SpatialProblem(
        objects=("N", "R", "S"),
        premise=And(
            (
                atom("S", Direction.SOUTHEAST, "R"),
                atom("N", Direction.SOUTHWEST, "S"),
            )
        ),
        query=DirectionQuery(
            "R",
            "N",
            frozenset({Direction.NORTHEAST, Direction.NORTHWEST}),
        ),
    )

    with pytest.raises(ProofConstructionError, match="non-refutable.*Northwest"):
        build_direction_answer_set(
            problem,
            {
                Direction.NORTHEAST: {
                    "N": (0, 0),
                    "R": (1, 2),
                    "S": (2, 1),
                }
            },
        )


def test_symbolic_direction_trace_rejects_a_tampered_domain() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=northeast,
        query=DirectionQuery("A", "B"),
    )
    certificate = build_direction_answer_set(
        problem,
        {Direction.NORTHEAST: {"A": (1, 1), "B": (0, 0)}},
    )
    payload = json.loads(
        render_direction_training_trace(certificate, TraceFormat.SYMBOLIC)
    )
    payload["possible_directions"] = ["SOUTHWEST"]

    with pytest.raises(ValueError, match="invalid unique Direction evidence"):
        parse_symbolic_direction_trace(problem, json.dumps(payload))


def test_builder_rejects_models_for_undeclared_candidates() -> None:
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=atom("A", Direction.NORTH, "B"),
        query=DirectionQuery("A", "B", frozenset({Direction.NORTH})),
    )

    with pytest.raises(ProofConstructionError, match="undeclared.*South"):
        build_direction_answer_set(
            problem,
            {
                Direction.NORTH: {"A": (0, 1), "B": (0, 0)},
                Direction.SOUTH: {"A": (0, -1), "B": (0, 0)},
            },
        )


def test_checker_rejects_incomplete_duplicate_and_mismatched_evidence() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=northeast,
        query=DirectionQuery("A", "B"),
    )
    certificate = build_direction_answer_set(
        problem,
        {Direction.NORTHEAST: {"A": (1, 1), "B": (0, 0)}},
    )

    with pytest.raises(AnswerSetCheckError, match="cover every declared direction"):
        check_direction_answer_set(
            replace(certificate, candidates=certificate.candidates[:-1])
        )

    duplicated = replace(
        certificate,
        candidates=(certificate.candidates[0], *certificate.candidates[:-1]),
    )
    with pytest.raises(AnswerSetCheckError, match="cover every declared direction"):
        check_direction_answer_set(duplicated)

    northeast_evidence = next(
        item for item in certificate.candidates if item.direction is Direction.NORTHEAST
    )
    southwest_refutation = next(
        item.evidence
        for item in certificate.candidates
        if item.direction is Direction.SOUTHWEST
    )
    assert isinstance(northeast_evidence.evidence, DirectionEntailmentCertificate)
    mismatched = DirectionCandidateCertificate(
        Direction.NORTHEAST,
        southwest_refutation,
    )
    mismatched_candidates = tuple(
        mismatched if item.direction is Direction.NORTHEAST else item
        for item in certificate.candidates
    )
    with pytest.raises(AnswerSetCheckError, match="refutation certifies the wrong"):
        check_direction_answer_set(
            DirectionAnswerSetCertificate(problem, mismatched_candidates)
        )

    assert isinstance(northeast_evidence.evidence, DirectionEntailmentCertificate)
    missing_proof = replace(
        certificate,
        candidates=tuple(
            replace(item, evidence=northeast_evidence.evidence.witness)
            if item.direction is Direction.NORTHEAST
            else item
            for item in certificate.candidates
        ),
    )
    with pytest.raises(AnswerSetCheckError, match="requires positive proof"):
        check_direction_answer_set(missing_proof)

    broken_entailment = replace(
        northeast_evidence.evidence,
        proof=replace(northeast_evidence.evidence.proof, conclusion_step="missing"),
    )
    broken = replace(
        certificate,
        candidates=tuple(
            replace(item, evidence=broken_entailment)
            if item.direction is Direction.NORTHEAST
            else item
            for item in certificate.candidates
        ),
    )
    with pytest.raises(AnswerSetCheckError, match="invalid entailment evidence"):
        check_direction_answer_set(broken)


def test_unique_direction_trace_requires_a_valid_consistency_witness() -> None:
    problem = SpatialProblem(
        ("A", "B"), atom("A", Direction.NORTH, "B"), DirectionQuery("A", "B")
    )
    certificate = build_direction_answer_set(
        problem, {Direction.NORTH: {"A": (0, 1), "B": (0, 0)}}
    )
    payload = json.loads(
        render_direction_training_trace(certificate, TraceFormat.SYMBOLIC)
    )
    witness = payload["evidence"][0].pop("witness")
    with pytest.raises(ValueError, match="unexpected or missing fields"):
        parse_symbolic_direction_trace(problem, json.dumps(payload))
    payload["evidence"][0]["witness"] = witness
    witness["y_order"].reverse()
    with pytest.raises(ValueError):
        parse_symbolic_direction_trace(problem, json.dumps(payload))


@pytest.mark.parametrize("query_kind", ("direction", "which", "count"))
def test_audit_has_numeric_witnesses_while_training_has_axis_orders(
    query_kind: str,
) -> None:
    from spatial.v2.answer_certificate_renderers import (
        render_audit_certificate,
        render_training_trace,
    )
    from spatial.v2.certificate_generation import build_answer_certificate
    from spatial.v2.solver import CountQuery, SpatialSolverV2, WhichQuery

    query = {
        "direction": DirectionQuery("A", "R"),
        "which": WhichQuery(frozenset({Direction.NORTH}), "R", ("A",)),
        "count": CountQuery(frozenset({Direction.NORTH}), "R", ("A",)),
    }[query_kind]
    problem = SpatialProblem(("A", "R"), atom("A", Direction.NORTH, "R"), query)
    solver = SpatialSolverV2("z3")
    certificate = build_answer_certificate(problem, solver.analyze(problem), solver)
    audit = render_audit_certificate(certificate, TraceFormat.SYMBOLIC)
    assert json.loads(audit)["schema"] == "spatial-audit-certificate-v1"
    assert '"assignments"' in audit and '"x":' in audit and '"y":' in audit
    natural_audit = render_audit_certificate(certificate, TraceFormat.NATURAL)
    assert "Coordinates:" in natural_audit
    for trace_format in TraceFormat:
        training = render_training_trace(certificate, trace_format)
        assert (
            "Coordinates:" not in training
            and '"x":' not in training
            and '"y":' not in training
        )
        assert (
            "X west to east:" if trace_format is TraceFormat.NATURAL else '"x_order"'
        ) in training
