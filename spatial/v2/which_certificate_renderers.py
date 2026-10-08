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
)
from spatial.v2.trace import TraceFormat
from spatial.v2.which_certificates import (
    MembershipEntailmentCertificate,
    MembershipImpossibilityCertificate,
    MembershipProofCertificate,
    WhichAnswerSetCertificate,
    check_which_answer_set,
)


def _render_refutations(
    certificate: MembershipEntailmentCertificate | MembershipImpossibilityCertificate,
    trace_format: TraceFormat,
    labels: Mapping[str, str],
) -> str:
    if not certificate.refutations:
        return "No direction exclusions are required."
    return "\n".join(
        render_direction_refutation(refutation, trace_format, labels)
        for refutation in certificate.refutations
    )


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
    lines = []
    for item in certificate.candidates:
        name = labels.get(item.candidate, item.candidate)
        if isinstance(item.evidence, MembershipProofCertificate):
            model = render_model_certificate(
                item.evidence.witness,
                trace_format,
                labels,
                include_coordinates=include_coordinates,
            )
            proof = render_direction_proof(
                item.evidence.proof,
                trace_format,
                labels,
            )
            evidence = f"{model}\n{proof}"
        elif isinstance(item.evidence, MembershipEntailmentCertificate):
            model = render_model_certificate(
                item.evidence.witness,
                trace_format,
                labels,
                include_coordinates=include_coordinates,
            )
            exclusions = _render_refutations(item.evidence, trace_format, labels)
            evidence = f"{model}\n{exclusions}"
        elif isinstance(item.evidence, MembershipImpossibilityCertificate):
            model = render_model_certificate(
                item.evidence.countermodel,
                trace_format,
                labels,
                include_coordinates=include_coordinates,
            )
            exclusions = _render_refutations(item.evidence, trace_format, labels)
            evidence = f"{model}\n{exclusions}"
        else:
            evidence = render_contingency_certificate(
                item.evidence,
                trace_format,
                labels,
                include_coordinates=include_coordinates,
            )
        lines.append(f"{name}: {item.status.value}\n{evidence}")

    possible = ", ".join(
        labels.get(candidate, candidate) for candidate in certificate.possible_entities
    )
    entailed = ", ".join(
        labels.get(candidate, candidate) for candidate in certificate.entailed_entities
    )
    if trace_format is TraceFormat.SYMBOLIC:
        lines.extend(
            (
                f"Which-Possible: {{{possible}}}",
                f"Which-Entailed: {{{entailed}}}",
            )
        )
    elif certificate.is_exact_single:
        lines.append(f"Therefore the unique entailed candidate is {entailed}.")
    elif possible:
        entailed_text = entailed or "none"
        lines.append(
            f"Therefore the possible candidates are {possible}; "
            f"the entailed candidates are {entailed_text}."
        )
    else:
        lines.append("Therefore no declared candidate can satisfy the query.")
    return "\n".join(lines)
