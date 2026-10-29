"""Versioned proof reuse for entailed Direction answers; legacy output is intact."""

from spatial.v2.answer_certificates import (
    DirectionAnswerSetCertificate,
    DirectionEntailmentCertificate,
)
from spatial.v2.composition_contracts import VerificationError, encode, parse_answer
from spatial.v2.composition_evidence import (
    _direction_text,
    verify_evidence,
)
from spatial.v2.composition_evidence import (
    render_pair as legacy_render_pair,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.solver import Direction, direction_signs

VERSION = "composition-entailed-direction-reuse-v1"


def exclusion_groups(direction):
    """Partition the other seven exact directions by a conflicting proved axis."""
    x, y = direction_signs(direction)
    return {
        "x": tuple(d.value for d in Direction if direction_signs(d)[0] != x),
        "y": tuple(
            d.value
            for d in Direction
            if direction_signs(d)[0] == x and direction_signs(d)[1] != y
        ),
    }


def render_pair(evidence, menu, mode, *, solver=None):
    """Reuse one checked entailment proof, without weakening full audit evidence."""
    mode = AnswerMode(mode)
    verify_evidence(evidence, solver=solver)
    semantics, certificate = evidence.semantics, evidence.certificate
    decision = encode(semantics, menu, mode)
    entailed = (
        [
            item.direction
            for item in certificate.candidates
            if isinstance(item.evidence, DirectionEntailmentCertificate)
            and item.direction.value in semantics.possible
        ]
        if isinstance(certificate, DirectionAnswerSetCertificate)
        else []
    )
    if (
        not semantics.consistent
        or semantics.case != "unique"
        or len(entailed) != 1
        or decision.escape_reason == "incomplete-coverage"
    ):
        return legacy_render_pair(
            evidence, menu, mode, solver=solver, compact_unique_direction=True
        )

    value = entailed[0].value
    groups = exclusion_groups(entailed[0])
    exclusions = [
        f"The proved {axis} comparison excludes {', '.join(values)}."
        for axis, values in groups.items()
        if values
    ]
    conclusion = f"The complete possible set is {{{value}}}."
    if decision.letters == "F":
        conclusion += (
            f" The unique answer {value} is absent from A-D."
            if mode is AnswerMode.SINGLE
            else " None of the displayed values is possible."
        )
    else:
        conclusion += f" It is displayed at {decision.letters}."
    pair = {
        style: "\n".join(
            [
                _direction_text(certificate, (value,), symbolic=style == "symbolic"),
                *exclusions,
                conclusion,
                f"Answer: {decision.letters}",
            ]
        )
        for style in ("natural", "symbolic")
    }
    for text in pair.values():
        if parse_answer(text, mode, stage=1) != decision.letters:
            raise VerificationError("compact proof changed the final answer")
    return pair
