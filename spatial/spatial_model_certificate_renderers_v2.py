"""Natural and symbolic renderings of checked spatial models."""

from __future__ import annotations

from collections.abc import Mapping

from spatial_explanation_renderers_v2 import TraceFormat
from spatial_model_certificates_v2 import (
    ContingencyCertificate,
    SpatialModelCertificate,
    check_contingency_certificate,
    check_model_certificate,
)
from spatial_proof_renderers_v2 import render_formula


def _coordinates(
    certificate: SpatialModelCertificate,
    labels: Mapping[str, str],
) -> str:
    return ", ".join(
        f"{labels.get(item.object, item.object)}=({item.x},{item.y})"
        for item in certificate.assignments
    )


def render_model_certificate(
    certificate: SpatialModelCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
    *,
    include_coordinates: bool = True,
) -> str:
    """Render one checked supporting or falsifying coordinate model."""
    check_model_certificate(certificate)
    trace_format = TraceFormat(trace_format)
    labels = labels or {}
    claim = render_formula(certificate.claim, trace_format, labels)
    value = "true" if certificate.expected_claim_value else "false"
    if trace_format is TraceFormat.SYMBOLIC:
        lines = ["Model: premises=true", f"Claim: {claim}={value}"]
        if include_coordinates:
            lines.insert(0, f"Coordinates: {{{_coordinates(certificate, labels)}}}")
        return "\n".join(lines)

    role = "supporting model" if certificate.expected_claim_value else "countermodel"
    lines = [f"This {role} satisfies every premise and makes the claim {value}."]
    if include_coordinates:
        lines.insert(0, f"Coordinates: {_coordinates(certificate, labels)}.")
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
