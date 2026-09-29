"""Coordinate-bearing reports for auditing V2 spatial analyses."""

from __future__ import annotations

from collections.abc import Mapping

from spatial_explanations_v2 import (
    AxisDerivation,
    ClaimEvidence,
    ClaimStatus,
    Coordinates,
    DirectionExplanation,
    QueryExplanation,
    WhichExplanation,
)
from spatial_solver_v2 import Direction


def _label(value: str, labels: Mapping[str, str]) -> str:
    return labels.get(value, value)


def _directions(directions: frozenset[Direction]) -> str:
    return ", ".join(
        direction.value for direction in Direction if direction in directions
    )


def _render_coordinates(
    coordinates: Coordinates,
    labels: Mapping[str, str],
) -> str:
    return ", ".join(
        f"{_label(obj, labels)}=({x}, {y})" for obj, (x, y) in coordinates.items()
    )


def _render_axis(derivation: AxisDerivation, labels: Mapping[str, str]) -> str:
    path = " -> ".join(_label(obj, labels) for obj in derivation.path)
    premises = ", ".join(str(index + 1) for index in derivation.premise_indices)
    subject = _label(derivation.subject, labels)
    reference = _label(derivation.reference, labels)
    return (
        f"On the {derivation.axis.value}-axis, premises {premises} give path "
        f"{path}; therefore {subject}.{derivation.axis.value} "
        f"{derivation.relation.value} {reference}.{derivation.axis.value}."
    )


def _render_evidence(
    subject: str,
    relation: str,
    reference: str,
    evidence: ClaimEvidence,
    labels: Mapping[str, str],
) -> list[str]:
    claim = f"{_label(subject, labels)} is {relation} of {_label(reference, labels)}"
    lines = [f"{claim}: {evidence.status.value}."]
    if evidence.axis_proof is not None:
        lines.extend(
            (
                _render_axis(evidence.axis_proof.x, labels),
                _render_axis(evidence.axis_proof.y, labels),
                f"Combining both axes gives {evidence.axis_proof.direction.value}.",
            )
        )
    elif evidence.negation_unsatisfiable:
        lines.append("Its negation is inconsistent with the premises.")
    if evidence.witness is not None and evidence.status is ClaimStatus.CONTINGENT:
        lines.append(
            f"Satisfying witness: {_render_coordinates(evidence.witness, labels)}."
        )
    if (
        evidence.counterexample is not None
        and evidence.status is ClaimStatus.CONTINGENT
    ):
        lines.append(
            f"Counterexample witness: {_render_coordinates(evidence.counterexample, labels)}."
        )
    return lines


def render_audit_explanation(
    explanation: QueryExplanation,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render coordinate witnesses for audits; do not use this as SFT CoT."""
    labels = labels or {}
    lines: list[str] = []
    if isinstance(explanation, DirectionExplanation):
        for case in explanation.cases:
            if case.evidence.status is ClaimStatus.IMPOSSIBLE:
                continue
            lines.extend(
                _render_evidence(
                    explanation.target,
                    case.direction.value,
                    explanation.reference,
                    case.evidence,
                    labels,
                )
            )
        impossible = [
            case.direction.value
            for case in explanation.cases
            if case.evidence.status is ClaimStatus.IMPOSSIBLE
        ]
        if impossible:
            lines.append("Impossible alternatives: " + ", ".join(impossible) + ".")
    elif isinstance(explanation, WhichExplanation):
        relation = _directions(explanation.directions)
        for membership in explanation.memberships:
            lines.extend(
                _render_evidence(
                    membership.candidate,
                    relation,
                    explanation.reference,
                    membership.evidence,
                    labels,
                )
            )
    else:
        lines.append(
            "Possible counts: " + ", ".join(map(str, explanation.possible_counts)) + "."
        )
        for case in explanation.counts:
            if case.witness is not None:
                lines.append(
                    f"Count {case.count} witness: {_render_coordinates(case.witness, labels)}."
                )
        relation = _directions(explanation.directions)
        for membership in explanation.memberships:
            lines.extend(
                _render_evidence(
                    membership.candidate,
                    relation,
                    explanation.reference,
                    membership.evidence,
                    labels,
                )
            )
    return "\n".join(lines)
