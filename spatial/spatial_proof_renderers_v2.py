"""Natural and symbolic renderings of checked SpatialEntail proof objects."""

from __future__ import annotations

from collections.abc import Mapping

from spatial_explanation_renderers_v2 import TraceFormat
from spatial_proofs_v2 import (
    AxisFact,
    DirectionClaim,
    DirectionProofCertificate,
    OrderRelation,
    ProofAxis,
    ProofRule,
    check_direction_proof,
)
from spatial_solver_v2 import RelationConstraint


def _label(value: str, labels: Mapping[str, str]) -> str:
    return labels.get(value, value)


def _direction_atom(
    atom: RelationConstraint,
    labels: Mapping[str, str],
) -> str:
    direction = next(iter(atom.allowed))
    return (
        f"{_label(atom.subject, labels)} is {direction.value} of "
        f"{_label(atom.reference, labels)}"
    )


def _axis_text(fact: AxisFact, labels: Mapping[str, str]) -> str:
    subject = _label(fact.subject, labels)
    reference = _label(fact.reference, labels)
    if fact.relation is OrderRelation.EQUAL:
        axis = "X" if fact.axis is ProofAxis.X else "Y"
        return f"{subject} and {reference} share the same {axis} coordinate"
    if fact.axis is ProofAxis.X:
        relation = "west" if fact.relation is OrderRelation.LESS else "east"
    else:
        relation = "south" if fact.relation is OrderRelation.LESS else "north"
    return f"{subject} is {relation} of {reference}"


def _axis_symbol(fact: AxisFact, labels: Mapping[str, str]) -> str:
    subject = _label(fact.subject, labels)
    reference = _label(fact.reference, labels)
    return f"{subject} {fact.relation.value}{fact.axis.value} {reference}"


def _claim_text(claim: DirectionClaim, labels: Mapping[str, str]) -> str:
    return (
        f"{_label(claim.subject, labels)} is {claim.direction.value} of "
        f"{_label(claim.reference, labels)}"
    )


def _natural_step(step, labels: Mapping[str, str]) -> str:
    conclusion = step.conclusion
    if step.rule is ProofRule.PREMISE:
        assert isinstance(conclusion, RelationConstraint)
        assert step.premise_index is not None
        return f"{step.id}: {_direction_atom(conclusion, labels)}."
    if step.rule is ProofRule.DIRECTION_DECOMPOSITION:
        assert isinstance(conclusion, AxisFact)
        return (
            f"{step.id}: From {step.inputs[0]} on the "
            f"{conclusion.axis.value.upper()}-axis, {_axis_text(conclusion, labels)}."
        )
    if step.rule is ProofRule.AXIS_INVERSION:
        assert isinstance(conclusion, AxisFact)
        return f"{step.id}: Equivalently, {_axis_text(conclusion, labels)}."
    if step.rule is ProofRule.AXIS_TRANSITIVITY:
        assert isinstance(conclusion, AxisFact)
        return (
            f"{step.id}: By {conclusion.axis.value.upper()}-axis transitivity from "
            f"{step.inputs[0]} and {step.inputs[1]}, {_axis_text(conclusion, labels)}."
        )
    assert step.rule is ProofRule.DIRECTION_RECOMPOSITION
    assert isinstance(conclusion, DirectionClaim)
    return (
        f"{step.id}: Combining the X and Y conclusions gives "
        f"{_claim_text(conclusion, labels)}."
    )


def _symbolic_step(step, labels: Mapping[str, str]) -> str:
    conclusion = step.conclusion
    if isinstance(conclusion, RelationConstraint):
        direction = next(iter(conclusion.allowed)).name
        rendered = (
            f"DIR_{direction}({_label(conclusion.subject, labels)},"
            f"{_label(conclusion.reference, labels)})"
        )
    elif isinstance(conclusion, AxisFact):
        rendered = _axis_symbol(conclusion, labels)
    else:
        rendered = (
            f"DIR_{conclusion.direction.name}({_label(conclusion.subject, labels)},"
            f"{_label(conclusion.reference, labels)})"
        )
    if step.rule is ProofRule.PREMISE:
        annotation = f"premise {step.premise_index + 1}"
    else:
        dependencies = ",".join(step.inputs)
        annotation = f"{step.rule.value} {dependencies}".rstrip()
    return f"{step.id}: {rendered}    [{annotation}]"


def render_direction_proof(
    certificate: DirectionProofCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render one checked proof object without consulting solver coordinates."""
    check_direction_proof(certificate)
    trace_format = TraceFormat(trace_format)
    labels = labels or {}
    if trace_format is TraceFormat.SYMBOLIC:
        return "\n".join(_symbolic_step(step, labels) for step in certificate.steps)

    return "\n".join(_natural_step(step, labels) for step in certificate.steps)
