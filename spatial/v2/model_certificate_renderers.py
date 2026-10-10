"""Natural and symbolic renderings of checked spatial models."""

from __future__ import annotations

from collections.abc import Mapping

from spatial.v2.model_certificates import (
    ContingencyCertificate,
    SpatialModelCertificate,
    check_contingency_certificate,
    check_model_certificate,
)
from spatial.v2.proof_renderers import render_formula
from spatial.v2.trace import TraceFormat


def _coordinates(
    certificate: SpatialModelCertificate,
    labels: Mapping[str, str],
) -> str:
    return ", ".join(
        f"{labels.get(item.object, item.object)}=({item.x},{item.y})"
        for item in certificate.assignments
    )


def _axis_order(
    certificate: SpatialModelCertificate,
    labels: Mapping[str, str],
    axis: str,
) -> str:
    coordinate = (lambda item: item.x) if axis == "x" else (lambda item: item.y)
    groups: dict[int, list[str]] = {}
    for assignment in certificate.assignments:
        groups.setdefault(coordinate(assignment), []).append(
            labels.get(assignment.object, assignment.object)
        )
    return " < ".join(" = ".join(sorted(groups[value])) for value in sorted(groups))


def render_model_certificate(
    certificate: SpatialModelCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
    *,
    include_coordinates: bool = True,
    claim_text: str | None = None,
) -> str:
    """Render one checked supporting or falsifying coordinate model."""
    check_model_certificate(certificate)
    trace_format = TraceFormat(trace_format)
    labels = labels or {}
    claim = claim_text or render_formula(certificate.claim, trace_format, labels)
    value = "true" if certificate.expected_claim_value else "false"
    if trace_format is TraceFormat.SYMBOLIC:
        lines = ["Model: premises=true", f"Claim: {claim}={value}"]
        if include_coordinates:
            lines.insert(0, f"Coordinates: {{{_coordinates(certificate, labels)}}}")
        else:
            lines[0:0] = (
                f"Order-X: {_axis_order(certificate, labels, 'x')}",
                f"Order-Y: {_axis_order(certificate, labels, 'y')}",
            )
        return "\n".join(lines)

    role = "supporting model" if certificate.expected_claim_value else "countermodel"
    lines = [
        f"This {role} satisfies every premise and makes the claim that {claim} {value}."
    ]
    if include_coordinates:
        lines.insert(0, f"Coordinates: {_coordinates(certificate, labels)}.")
    else:
        lines[0:0] = (
            f"X west to east: {_axis_order(certificate, labels, 'x')}.",
            f"Y south to north: {_axis_order(certificate, labels, 'y')}.",
        )
    return "\n".join(lines)


def render_contingency_certificate(
    certificate: ContingencyCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
    *,
    include_coordinates: bool = True,
) -> str:
    """Render two checked models showing that a claim is contingent."""
    check_contingency_certificate(certificate)
    trace_format = TraceFormat(trace_format)
    labels = labels or {}
    separator = "\n"
    if trace_format is TraceFormat.SYMBOLIC:
        heading = "Status: contingent"
        witness_label = "Witness"
        counterexample_label = "Counterexample"
    else:
        heading = "The claim is possible but not entailed."
        witness_label = "Supporting case"
        counterexample_label = "Counterexample"
    return separator.join(
        (
            heading,
            f"{witness_label}:\n"
            + render_model_certificate(
                certificate.witness,
                trace_format,
                labels,
                include_coordinates=include_coordinates,
            ),
            f"{counterexample_label}:\n"
            + render_model_certificate(
                certificate.counterexample,
                trace_format,
                labels,
                include_coordinates=include_coordinates,
            ),
        )
    )
