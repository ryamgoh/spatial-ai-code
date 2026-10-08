"""Natural and symbolic renderings of complete Count evidence."""

from __future__ import annotations

from collections.abc import Mapping

from spatial_count_certificates_v2 import (
    CountAnswerSetCertificate,
    CountImpossibilityCertificate,
    check_count_answer_set,
)
from spatial_explanation_renderers_v2 import TraceFormat
from spatial_model_certificate_renderers_v2 import render_model_certificate
from spatial_proof_renderers_v2 import render_formula_refutation


def render_count_answer_set(
    certificate: CountAnswerSetCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
    *,
    include_coordinates: bool = True,
) -> str:
    """Render complete evidence for every candidate count."""
    check_count_answer_set(certificate)
    trace_format = TraceFormat(trace_format)
    labels = labels or {}
    lines = []
    for item in certificate.values:
        if isinstance(item.evidence, CountImpossibilityCertificate):
            assignment_lines = []
            for assignment in item.evidence.assignments:
                members = ", ".join(
                    labels.get(member, member) for member in assignment.members
                )
                assignment_lines.append(
                    f"membership assignment {{{members}}}\n"
                    + render_formula_refutation(
                        assignment.refutation,
                        trace_format,
                        labels,
                    )
                )
            evidence = "\n".join(assignment_lines)
            status = "impossible"
        else:
            evidence = render_model_certificate(
                item.evidence,
                trace_format,
                labels,
                include_coordinates=include_coordinates,
            )
            status = "possible"
        lines.append(f"Count {item.count}: {status}\n{evidence}")

    possible = ", ".join(str(count) for count in certificate.possible_counts)
    if trace_format is TraceFormat.SYMBOLIC:
        lines.append(f"Count-Domain: {{{possible}}}")
    elif certificate.is_unique:
        lines.append(f"Therefore the unique count is {possible}.")
    else:
        lines.append(f"Therefore the possible counts are {possible}.")
    return "\n".join(lines)
