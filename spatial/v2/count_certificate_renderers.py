"""Natural and symbolic renderings of complete Count evidence."""

from __future__ import annotations

from collections.abc import Mapping

from spatial.v2.count_certificates import (
    CountAnswerSetCertificate,
    CountImpossibilityCertificate,
    CountValueCertificate,
    check_count_answer_set,
)
from spatial.v2.model_certificate_renderers import render_model_certificate
from spatial.v2.proof_renderers import render_formula_refutation
from spatial.v2.solver import CountQuery, Direction
from spatial.v2.symbolic_trace_codec import render_symbolic_count_trace
from spatial.v2.trace import TraceFormat
from spatial.v2.which_certificate_renderers import render_membership_evidence


def _render_count_evidence(
    item: CountValueCertificate,
    trace_format: TraceFormat,
    labels: Mapping[str, str],
    *,
    include_coordinates: bool,
) -> tuple[str, str]:
    if isinstance(item.evidence, CountImpossibilityCertificate):
        assignment_lines = []
        for assignment in item.evidence.assignments:
            members = ", ".join(
                labels.get(member, member) for member in assignment.members
            )
            assignment_lines.append(
                f"membership assignment {{{members}}}\n"
                + render_formula_refutation(assignment.refutation, trace_format, labels)
            )
        return (
            "\n".join(assignment_lines)
            or "No joint assignment of this size agrees with the fixed memberships.",
            "impossible",
        )
    query = item.evidence.problem.query
    assert isinstance(query, CountQuery)
    candidates = ", ".join(labels.get(name, name) for name in query.candidates)
    directions = ", ".join(
        direction.value for direction in Direction if direction in query.directions
    )
    reference = labels.get(query.reference, query.reference)
    return (
        render_model_certificate(
            item.evidence,
            trace_format,
            labels,
            include_coordinates=include_coordinates,
            claim_text=f"exactly {item.count} of {{{candidates}}} are in {{{directions}}} of {reference}",
        ),
        "possible",
    )


def _fixed_membership_lines(
    certificate: CountAnswerSetCertificate,
    trace_format: TraceFormat,
    labels: Mapping[str, str],
    *,
    include_coordinates: bool,
    training: bool,
) -> list[str]:
    lines = []
    for item in certificate.fixed_memberships:
        lines.append(
            render_membership_evidence(
                item,
                trace_format,
                labels,
                include_coordinates=include_coordinates,
                training=training,
            )
        )
        lines.append(
            f"Fixed membership: {labels.get(item.candidate, item.candidate)} is {item.status.value}."
        )
    if certificate.fixed_memberships:
        lines.append(
            "Every joint assignment must include the entailed candidates and exclude the impossible candidates; the remaining candidates are checked jointly below."
        )
    return lines


def _count_summary(
    certificate: CountAnswerSetCertificate,
    trace_format: TraceFormat,
) -> str:
    possible = ", ".join(str(count) for count in certificate.possible_counts)
    if trace_format is TraceFormat.SYMBOLIC:
        return f"Count-Domain: {{{possible}}}"
    if certificate.is_unique:
        return f"Therefore the unique count is {possible}."
    return f"Therefore the possible counts are {possible}."


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
    if trace_format is TraceFormat.SYMBOLIC and include_coordinates:
        from spatial.v2.answer_certificate_renderers import render_audit_certificate

        return render_audit_certificate(certificate, trace_format, labels)
    if trace_format is TraceFormat.SYMBOLIC:
        if labels:
            raise ValueError(
                "symbolic Count traces use certificate identifiers and do not "
                "accept display labels"
            )
        return render_symbolic_count_trace(certificate)
    lines = _fixed_membership_lines(
        certificate,
        trace_format,
        labels,
        include_coordinates=include_coordinates,
        training=False,
    )
    for item in certificate.values:
        evidence, status = _render_count_evidence(
            item,
            trace_format,
            labels,
            include_coordinates=include_coordinates,
        )
        lines.append(f"Count {item.count}: {status}\n{evidence}")
    lines.append(_count_summary(certificate, trace_format))
    return "\n".join(lines)


def render_count_training_trace(
    certificate: CountAnswerSetCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render derivation-first Count supervision without numeric coordinates."""
    check_count_answer_set(certificate)
    trace_format = TraceFormat(trace_format)
    labels = labels or {}
    if trace_format is TraceFormat.SYMBOLIC:
        if labels:
            raise ValueError(
                "symbolic Count traces use certificate identifiers and do not "
                "accept display labels"
            )
        return render_symbolic_count_trace(certificate)
    lines = _fixed_membership_lines(
        certificate, trace_format, labels, include_coordinates=False, training=True
    )
    for item in certificate.values:
        evidence, status = _render_count_evidence(
            item,
            trace_format,
            labels,
            include_coordinates=False,
        )
        lines.extend((evidence, f"Count {item.count}: {status}"))
    lines.append(_count_summary(certificate, trace_format))
    return "\n".join(lines)
