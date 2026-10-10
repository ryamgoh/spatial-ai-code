"""Natural and symbolic renderings of complete Direction answer evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

from spatial.v2.answer_certificates import (
    DirectionAnswerSetCertificate,
    DirectionEntailmentCertificate,
    check_direction_answer_set,
)
from spatial.v2.count_certificate_renderers import (
    render_count_answer_set,
    render_count_training_trace,
)
from spatial.v2.count_certificates import (
    CountAnswerSetCertificate,
    check_count_answer_set,
)
from spatial.v2.model_certificate_renderers import render_model_certificate
from spatial.v2.model_certificates import SpatialModelCertificate
from spatial.v2.proof_renderers import (
    render_direction_proof,
    render_direction_refutation,
    render_direction_training_proof,
    render_direction_training_refutation,
)
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.symbolic_trace_codec import render_symbolic_direction_trace
from spatial.v2.trace import TraceFormat
from spatial.v2.which_certificate_renderers import (
    render_which_answer_set,
    render_which_training_trace,
)
from spatial.v2.which_certificates import (
    WhichAnswerSetCertificate,
    check_which_answer_set,
)

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
    if trace_format is TraceFormat.SYMBOLIC and include_coordinates:
        return render_audit_certificate(certificate, trace_format, labels)
    lines = []
    for item in certificate.candidates:
        if isinstance(item.evidence, DirectionEntailmentCertificate):
            evidence = "\n".join(
                (
                    render_model_certificate(
                        item.evidence.witness,
                        trace_format,
                        labels,
                        include_coordinates=include_coordinates,
                    ),
                    render_direction_proof(
                        item.evidence.proof,
                        trace_format,
                        labels,
                    ),
                )
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
    elif certificate.is_unique:
        lines.append(f"Therefore the unique direction is {possible}.")
    else:
        lines.append(f"Therefore the possible directions are {possible}.")
    return "\n".join(lines)


def render_direction_training_trace(
    certificate: DirectionAnswerSetCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render compact Direction supervision while retaining the full audit object."""
    check_direction_answer_set(certificate)
    trace_format = TraceFormat(trace_format)
    labels = labels or {}
    if trace_format is TraceFormat.SYMBOLIC:
        if labels:
            raise ValueError(
                "symbolic Direction traces use certificate identifiers and do not "
                "accept display labels"
            )
        return render_symbolic_direction_trace(certificate)
    if not certificate.is_unique:
        lines = []
        for item in certificate.candidates:
            if isinstance(item.evidence, SpatialModelCertificate):
                evidence = render_model_certificate(
                    item.evidence,
                    trace_format,
                    labels,
                    include_coordinates=False,
                )
                status = "possible"
            else:
                evidence = render_direction_training_refutation(
                    item.evidence,
                    labels,
                )
                status = "impossible"
            lines.extend((evidence, f"{item.direction.value}: {status}"))
        possible = ", ".join(
            direction.value for direction in certificate.possible_directions
        )
        lines.append(
            f"Direction-Domain: {{{possible}}}"
            if trace_format is TraceFormat.SYMBOLIC
            else f"Therefore the possible directions are {possible}."
        )
        return "\n".join(lines)

    entailed = next(item for item in certificate.candidates if item.entailed)
    assert isinstance(entailed.evidence, DirectionEntailmentCertificate)
    proof = entailed.evidence.proof
    lines = [
        render_direction_training_proof(proof, labels),
        render_model_certificate(
            entailed.evidence.witness, trace_format, labels, include_coordinates=False
        ),
    ]
    lines.append(
        f"Therefore the unique direction is {entailed.direction.value}; "
        "all other exact directions are excluded because "
        "exact directions are mutually exclusive."
    )
    return "\n".join(lines)


def render_audit_certificate(
    certificate: AnswerCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render exhaustive checked evidence, including constructive coordinates."""
    if TraceFormat(trace_format) is TraceFormat.SYMBOLIC:
        if labels:
            raise ValueError("symbolic audit certificates use certificate identifiers")
        if isinstance(certificate, DirectionAnswerSetCertificate):
            check_direction_answer_set(certificate)
        elif isinstance(certificate, WhichAnswerSetCertificate):
            check_which_answer_set(certificate)
        elif isinstance(certificate, CountAnswerSetCertificate):
            check_count_answer_set(certificate)
        else:
            raise TypeError(
                f"unsupported answer certificate: {type(certificate).__name__}"
            )
        return json.dumps(
            {
                "schema": "spatial-audit-certificate-v1",
                "certificate": tagged_dataclass_to_dict(certificate),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    if isinstance(certificate, DirectionAnswerSetCertificate):
        return render_direction_answer_set(
            certificate,
            trace_format,
            labels,
            include_coordinates=True,
        )
    if isinstance(certificate, WhichAnswerSetCertificate):
        return render_which_answer_set(
            certificate,
            trace_format,
            labels,
            include_coordinates=True,
        )
    if isinstance(certificate, CountAnswerSetCertificate):
        return render_count_answer_set(
            certificate,
            trace_format,
            labels,
            include_coordinates=True,
        )
    raise TypeError(f"unsupported answer certificate: {type(certificate).__name__}")


def render_training_trace(
    certificate: AnswerCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render compact supervision without presenting coordinates as ground truth."""
    if isinstance(certificate, DirectionAnswerSetCertificate):
        return render_direction_training_trace(certificate, trace_format, labels)
    if isinstance(certificate, WhichAnswerSetCertificate):
        return render_which_training_trace(certificate, trace_format, labels)
    if isinstance(certificate, CountAnswerSetCertificate):
        return render_count_training_trace(certificate, trace_format, labels)
    raise TypeError(f"unsupported answer certificate: {type(certificate).__name__}")
