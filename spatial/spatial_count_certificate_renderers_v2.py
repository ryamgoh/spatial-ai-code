"""Natural and symbolic renderings of complete Count evidence."""

from __future__ import annotations

from collections.abc import Mapping

from spatial_count_certificates_v2 import (
    CountAnswerSetCertificate,
    CountImpossibilityCertificate,
    CountMembershipConflict,
    check_count_answer_set,
)
from spatial_model_certificate_renderers_v2 import render_model_certificate
from spatial_proof_renderers_v2 import (
    render_direction_proof,
    render_direction_refutation,
    render_formula_refutation,
)
from spatial_trace_v2 import TraceFormat
from spatial_which_certificates_v2 import (
    MembershipEntailmentCertificate,
    MembershipImpossibilityCertificate,
    MembershipProofCertificate,
)


def _render_membership_conflict(
    conflict: CountMembershipConflict,
    trace_format: TraceFormat,
    labels: Mapping[str, str],
) -> str:
    evidence = conflict.evidence
    if isinstance(evidence, MembershipProofCertificate):
        return render_direction_proof(evidence.proof, trace_format, labels)
    if isinstance(evidence, MembershipEntailmentCertificate):
        return "\n".join(
            render_direction_refutation(refutation, trace_format, labels)
            for refutation in evidence.refutations
        )
    assert isinstance(evidence, MembershipImpossibilityCertificate)
    return "\n".join(
        render_direction_refutation(refutation, trace_format, labels)
        for refutation in evidence.refutations
    )


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
                    + (
                        _render_membership_conflict(
                            assignment.refutation,
                            trace_format,
                            labels,
                        )
                        if isinstance(
                            assignment.refutation,
                            CountMembershipConflict,
                        )
                        else render_formula_refutation(
                            assignment.refutation,
                            trace_format,
                            labels,
                        )
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
