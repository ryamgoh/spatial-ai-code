"""Natural and symbolic renderings of complete Direction answer evidence."""

from __future__ import annotations

from collections.abc import Mapping

from spatial_answer_certificates_v2 import (
    DirectionAnswerSetCertificate,
    check_direction_answer_set,
)
from spatial_explanation_renderers_v2 import TraceFormat
from spatial_model_certificate_renderers_v2 import render_model_certificate
from spatial_model_certificates_v2 import SpatialModelCertificate
from spatial_proof_renderers_v2 import render_direction_refutation


def render_direction_answer_set(
    certificate: DirectionAnswerSetCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
    *,
    include_coordinates: bool = True,
) -> str:
    """Render exhaustive candidate evidence and the resulting possibility set."""
    check_direction_answer_set(certificate)
    trace_format = TraceFormat(trace_format)
    labels = labels or {}
    lines = []
    for item in certificate.candidates:
        if isinstance(item.evidence, SpatialModelCertificate):
            evidence = render_model_certificate(
                item.evidence,
                trace_format,
                labels,
                include_coordinates=include_coordinates,
            )
            status = "possible"
        else:
            evidence = render_direction_refutation(
                item.evidence,
                trace_format,
                labels,
            )
            status = "impossible"
        lines.append(f"{item.direction.value}: {status}\n{evidence}")

    possible = ", ".join(
        direction.value for direction in certificate.possible_directions
    )
    if trace_format is TraceFormat.SYMBOLIC:
        lines.append(f"Direction-Domain: {{{possible}}}")
    elif certificate.is_unique:
        lines.append(f"Therefore the unique direction is {possible}.")
    else:
        lines.append(f"Therefore the possible directions are {possible}.")
    return "\n".join(lines)
