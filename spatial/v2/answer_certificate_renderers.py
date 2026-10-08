"""Natural and symbolic renderings of complete Direction answer evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from spatial.v2.answer_certificates import (
    DirectionAnswerSetCertificate,
    DirectionEntailmentCertificate,
    check_direction_answer_set,
)
from spatial.v2.count_certificate_renderers import render_count_answer_set
from spatial.v2.count_certificates import CountAnswerSetCertificate
from spatial.v2.model_certificate_renderers import render_model_certificate
from spatial.v2.model_certificates import SpatialModelCertificate
from spatial.v2.proof_renderers import (
    render_direction_proof,
    render_direction_refutation,
)
from spatial.v2.trace import TraceFormat
from spatial.v2.which_certificate_renderers import render_which_answer_set
from spatial.v2.which_certificates import WhichAnswerSetCertificate

if TYPE_CHECKING:
    from spatial.v2.certificate_generation import AnswerCertificate


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
    if certificate.is_unique:
        entailed = next(item for item in certificate.candidates if item.entailed)
        assert isinstance(entailed.evidence, DirectionEntailmentCertificate)
        lines = [
            f"{entailed.direction.value}: entailed\n"
            + render_direction_proof(
                entailed.evidence.proof,
                trace_format,
                labels,
            )
        ]
        for item in certificate.candidates:
            if item is entailed:
                continue
            if trace_format is TraceFormat.SYMBOLIC:
                lines.append(
                    f"{item.direction.value}: impossible "
                    f"[exclusive-with={entailed.direction.value}]"
                )
            else:
                lines.append(
                    f"{item.direction.value} is impossible because it is mutually "
                    f"exclusive with the proved {entailed.direction.value} direction."
                )
        lines.append(
            f"Direction-Domain: {{{entailed.direction.value}}}"
            if trace_format is TraceFormat.SYMBOLIC
            else f"Therefore the unique direction is {entailed.direction.value}."
        )
        return "\n".join(lines)

    lines = []
    for item in certificate.candidates:
        if isinstance(item.evidence, DirectionEntailmentCertificate):
            evidence = render_direction_proof(
                item.evidence.proof,
                trace_format,
                labels,
            )
            status = "entailed"
        elif isinstance(item.evidence, SpatialModelCertificate):
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
    else:
        lines.append(f"Therefore the possible directions are {possible}.")
    return "\n".join(lines)


def render_answer_certificate(
    certificate: AnswerCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
    *,
    include_coordinates: bool = True,
) -> str:
    """Render any checked Direction, Which, or Count answer certificate."""
    if isinstance(certificate, DirectionAnswerSetCertificate):
        return render_direction_answer_set(
            certificate,
            trace_format,
            labels,
            include_coordinates=include_coordinates,
        )
    if isinstance(certificate, WhichAnswerSetCertificate):
        return render_which_answer_set(
            certificate,
            trace_format,
            labels,
            include_coordinates=include_coordinates,
        )
    if isinstance(certificate, CountAnswerSetCertificate):
        return render_count_answer_set(
            certificate,
            trace_format,
            labels,
            include_coordinates=include_coordinates,
        )
    raise TypeError(f"unsupported answer certificate: {type(certificate).__name__}")
