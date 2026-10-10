"""Natural and symbolic renderings of complete Which membership evidence."""

from __future__ import annotations

from collections.abc import Mapping

from spatial.v2.model_certificate_renderers import (
    render_contingency_certificate,
    render_model_certificate,
)
from spatial.v2.proof_renderers import (
    render_direction_proof,
    render_direction_refutation,
    render_direction_training_proof,
    render_direction_training_refutation,
)
from spatial.v2.symbolic_trace_codec import render_symbolic_which_trace
from spatial.v2.trace import TraceFormat
from spatial.v2.which_certificates import (
    MembershipEntailmentCertificate,
    MembershipImpossibilityCertificate,
    MembershipProofCertificate,
    WhichAnswerSetCertificate,
    WhichCandidateCertificate,
    check_which_answer_set,
)


def _render_refutations(
    certificate: MembershipEntailmentCertificate | MembershipImpossibilityCertificate,
    trace_format: TraceFormat,
    labels: Mapping[str, str],
    *,
    training: bool,
) -> str:
    if not certificate.refutations:
        return "No direction exclusions are required."
    return "\n".join(
        (
            render_direction_training_refutation(refutation, labels)
            if training
            else render_direction_refutation(refutation, trace_format, labels)
        )
        for refutation in certificate.refutations
    )


def render_membership_evidence(
    item: WhichCandidateCertificate,
    trace_format: TraceFormat,
    labels: Mapping[str, str],
    *,
    include_coordinates: bool,
    training: bool,
) -> str:
    evidence = item.evidence
    if isinstance(evidence, MembershipProofCertificate):
        proof = (
            render_direction_training_proof(evidence.proof, labels)
            if training
            else render_direction_proof(evidence.proof, trace_format, labels)
        )
        return "\n".join(
            (
                render_model_certificate(
                    evidence.witness,
                    trace_format,
                    labels,
                    include_coordinates=include_coordinates,
                ),
                proof,
            )
        )
    if isinstance(
        evidence,
        (MembershipEntailmentCertificate, MembershipImpossibilityCertificate),
    ):
        model = (
            evidence.witness
            if isinstance(evidence, MembershipEntailmentCertificate)
            else evidence.countermodel
        )
        return "\n".join(
            (
                render_model_certificate(
                    model,
                    trace_format,
                    labels,
                    include_coordinates=include_coordinates,
                ),
                _render_refutations(
                    evidence,
                    trace_format,
                    labels,
                    training=training,
                ),
            )
        )
    return render_contingency_certificate(
        evidence,
        trace_format,
        labels,
        include_coordinates=include_coordinates,
    )


def _which_summary(
    certificate: WhichAnswerSetCertificate,
    labels: Mapping[str, str],
) -> str:
    possible = ", ".join(
        labels.get(candidate, candidate) for candidate in certificate.possible_entities
    )
    entailed = ", ".join(
        labels.get(candidate, candidate) for candidate in certificate.entailed_entities
    )
    if certificate.is_exact_single:
        return f"Therefore the unique entailed candidate is {entailed}."
    if possible:
        return (
            f"Therefore the possible candidates are {possible}; "
            f"the entailed candidates are {entailed or 'none'}."
        )
    return "Therefore no declared candidate can satisfy the query."


def render_which_answer_set(
    certificate: WhichAnswerSetCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
    *,
    include_coordinates: bool = True,
) -> str:
    """Render every checked candidate status and the resulting Which domains."""
    check_which_answer_set(certificate)
    trace_format = TraceFormat(trace_format)
    labels = labels or {}
    if trace_format is TraceFormat.SYMBOLIC and include_coordinates:
        from spatial.v2.answer_certificate_renderers import render_audit_certificate

        return render_audit_certificate(certificate, trace_format, labels)
    if trace_format is TraceFormat.SYMBOLIC:
        if labels:
            raise ValueError(
                "symbolic Which traces use certificate identifiers and do not "
                "accept display labels"
            )
        return render_symbolic_which_trace(certificate)
    lines = []
    for item in certificate.candidates:
        name = labels.get(item.candidate, item.candidate)
        evidence = render_membership_evidence(
            item,
            trace_format,
            labels,
            include_coordinates=include_coordinates,
            training=False,
        )
        lines.append(f"{name}: {item.status.value}\n{evidence}")
    lines.append(_which_summary(certificate, labels))
    return "\n".join(lines)


def render_which_training_trace(
    certificate: WhichAnswerSetCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render derivation-first Which supervision without numeric coordinates."""
    check_which_answer_set(certificate)
    trace_format = TraceFormat(trace_format)
    labels = labels or {}
    if trace_format is TraceFormat.SYMBOLIC:
        if labels:
            raise ValueError(
                "symbolic Which traces use certificate identifiers and do not "
                "accept display labels"
            )
        return render_symbolic_which_trace(certificate)
    lines = []
    for item in certificate.candidates:
        name = labels.get(item.candidate, item.candidate)
        evidence = render_membership_evidence(
            item,
            trace_format,
            labels,
            include_coordinates=False,
            training=True,
        )
        lines.extend((evidence, f"{name}: {item.status.value}"))
    lines.append(_which_summary(certificate, labels))
    return "\n".join(lines)
