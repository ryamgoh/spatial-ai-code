"""First composition-v1 evidence path: provable, consistent Direction queries.

This is a feasibility path, not a dataset release. Ambiguity, Selection, Count,
and conflict explanations remain unsupported here, even when labels are known.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from spatial.v2.composition_contracts import (
    VERSION,
    Semantics,
    UnsupportedEvidence,
    VerificationError,
    analyze,
    check_witness,
    encode,
    parse_answer,
    parse_prompt,
    render_prompt,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.proofs import (
    AxisFact,
    DirectionClaim,
    DirectionProofCertificate,
    ProofConstructionError,
    ProofRule,
    build_direction_proof,
    check_direction_proof,
    proof_to_dict,
)
from spatial.v2.solver import (
    And,
    Direction,
    DirectionQuery,
    RelationConstraint,
    SpatialProblem,
    SpatialSolverV2,
)

_SUPPORTED_RULES = {
    ProofRule.PREMISE,
    ProofRule.DIRECTION_DECOMPOSITION,
    ProofRule.AXIS_INVERSION,
    ProofRule.AXIS_TRANSITIVITY,
    ProofRule.ATOM_RECOMPOSITION,
    ProofRule.DIRECTION_INTRODUCTION,
    ProofRule.DIRECTION_RECOMPOSITION,
}


@dataclass(frozen=True)
class DirectionEvidence:
    semantics: Semantics
    proof: DirectionProofCertificate


def build_evidence(
    problem: SpatialProblem, *, solver: SpatialSolverV2 | None = None
) -> DirectionEvidence:
    semantics = analyze(problem, solver=solver)
    if not isinstance(problem.query, DirectionQuery) or semantics.case != "unique":
        raise UnsupportedEvidence(
            f"Direction proof path does not support {type(problem.query).__name__}/"
            f"{semantics.case}; do not release a label-only synthetic row"
        )
    direction = Direction(next(iter(semantics.possible)))
    try:
        proof = build_direction_proof(problem, direction)
    except ProofConstructionError as exc:
        raise UnsupportedEvidence(
            "no supported axis derivation for this answer"
        ) from exc
    if any(step.rule not in _SUPPORTED_RULES for step in proof.steps):
        raise UnsupportedEvidence("proof requires a rule outside the atomic renderer")
    return DirectionEvidence(semantics, proof)


def verify_evidence(
    evidence: DirectionEvidence, *, solver: SpatialSolverV2 | None = None
) -> None:
    """Replay evidence and recheck its semantic conclusion before rendering."""
    semantics, proof = evidence.semantics, evidence.proof
    if proof.problem != semantics.problem:
        raise VerificationError("proof and semantic record refer to different problems")
    check_direction_proof(proof)
    if any(step.rule not in _SUPPORTED_RULES for step in proof.steps):
        raise UnsupportedEvidence("unsupported proof rule")
    actual = analyze(semantics.problem, solver=solver)
    if (
        actual.case != "unique"
        or not actual.consistent
        or semantics.consistent != actual.consistent
        or semantics.possible != actual.possible
        or semantics.certain != actual.certain
        or proof.conclusion.direction.value not in actual.possible
    ):
        raise VerificationError("evidence disagrees with full-domain solver analysis")
    if set(semantics.witnesses) != semantics.possible:
        raise VerificationError("stored witnesses do not cover the possible set")
    for value, coordinates in semantics.witnesses.items():
        check_witness(semantics.problem, coordinates, value)


def _fact_text(fact, *, symbolic: bool) -> str:
    if isinstance(fact, AxisFact):
        axis = fact.axis.value
        if symbolic:
            return (
                f"{axis}({fact.subject}) {fact.relation.value} {axis}({fact.reference})"
            )
        relation = {"=": "equal to", "<": "less than", ">": "greater than"}[
            fact.relation.value
        ]
        return (
            f"{fact.subject}'s {axis}-coordinate is {relation} "
            f"{fact.reference}'s {axis}-coordinate"
        )
    if isinstance(fact, (RelationConstraint, DirectionClaim)):
        direction = (
            next(iter(fact.allowed))
            if isinstance(fact, RelationConstraint)
            else fact.direction
        )
        if symbolic:
            return f"{direction.value}({fact.subject},{fact.reference})"
        return (
            f"{fact.subject} is exactly {direction.value.lower()} of {fact.reference}"
        )
    raise UnsupportedEvidence(f"cannot render {type(fact).__name__}")


def _render_body(evidence: DirectionEvidence, *, symbolic: bool) -> str:
    # Both styles traverse precisely the same checked steps, including rewrites.
    lines = []
    for step in evidence.proof.steps:
        conclusion = _fact_text(step.conclusion, symbolic=symbolic)
        if step.rule is ProofRule.PREMISE:
            source = f"p{step.premise_index + 1}"
        else:
            source = ", ".join(step.inputs)
        if symbolic:
            lines.append(f"{step.id}: [{source}] => {conclusion}")
        else:
            lines.append(f"{step.id}: From {source}, {conclusion}.")
    direction = evidence.proof.conclusion.direction.value
    lines.append(
        f"possible={{{direction}}}; the other seven exact directions are excluded."
        if symbolic
        else f"Thus {direction} is the direction in every valid map. "
        "The other seven exact directions are excluded by these coordinate comparisons."
    )
    return "\n".join(lines)


def render_pair(
    evidence: DirectionEvidence,
    menu: tuple[str, ...],
    mode: AnswerMode,
    *,
    solver: SpatialSolverV2 | None = None,
) -> dict[str, str]:
    verify_evidence(evidence, solver=solver)
    decision = encode(evidence.semantics, menu, mode)
    value = next(iter(evidence.semantics.possible))
    if decision.letters == "F":
        conclusion = {
            "unique-unlisted": f"The unique answer {value} is not displayed in A-D.",
            "no-listed-possibilities": "None of the displayed A-D values is possible.",
            "incomplete-coverage": f"The possible value {value} is missing from A-D, so no complete answer can be formed.",
        }[decision.escape_reason]
    else:
        conclusion = f"The complete possible set is {{{value}}}, displayed at {decision.letters}."
    pair = {
        style: _render_body(evidence, symbolic=style == "symbolic")
        + f"\n{conclusion}\nAnswer: {decision.letters}"
        for style in ("natural", "symbolic")
    }
    for text in pair.values():
        if parse_answer(text, mode, stage=1) != decision.letters:
            raise VerificationError("rendered explanation has an invalid final answer")
    return pair


def build_example(
    problem: SpatialProblem,
    menu: tuple[str, ...],
    mode: AnswerMode,
    *,
    story_id: str,
    family_id: str,
) -> dict:
    """Build an auditable feasibility example, never assign a dataset split."""
    if not story_id or not family_id:
        raise ValueError("story_id and family_id are required")
    mode = AnswerMode(mode)
    evidence = build_evidence(problem)
    prompt = render_prompt(problem, menu, mode)
    if parse_prompt(prompt) != (problem, menu, mode):
        raise VerificationError("full prompt round trip changed the problem")
    pair = render_pair(evidence, menu, mode)
    semantics = evidence.semantics
    decision = encode(semantics, menu, mode)
    return {
        "interface": VERSION,
        "status": "feasibility-only-not-a-dataset-release",
        "story_id": story_id,
        "family_id": family_id,
        "contract": mode.value,
        "prompt": prompt,
        "references": pair,
        "answer": decision.letters,
        "audit": {
            "semantic_case": semantics.case,
            "escape_reason": decision.escape_reason,
            "consistent": semantics.consistent,
            "possible": sorted(semantics.possible),
            "excluded": [
                d.value for d in Direction if d.value not in semantics.possible
            ],
            "solver_engine": semantics.engine,
            "witnesses": semantics.witnesses,
            "proof": proof_to_dict(evidence.proof),
            "support_premise_ids": [
                f"p{index + 1}" for index in evidence.proof.support_premise_indices
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    problem = SpatialProblem(
        ("X", "Y", "Z"),
        And(
            (
                RelationConstraint("X", "Y", frozenset({Direction.NORTHEAST})),
                RelationConstraint("Z", "Y", frozenset({Direction.SOUTH})),
            )
        ),
        DirectionQuery("X", "Z"),
    )
    menus = (
        ("West", "Northeast", "South", "Northwest"),
        ("North", "East", "South", "West"),
    )
    examples = [
        build_example(problem, menu, mode, story_id="D1", family_id="D1")
        for menu in menus
        for mode in AnswerMode
    ]
    payload = {
        "status": "feasibility-only-not-a-dataset-release",
        "unique_stories": 1,
        "views": len(examples),
        "examples": examples,
        "unsupported_evidence": [
            "ambiguous-direction",
            "selection",
            "count",
            "inconsistency",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(examples)} checked D1 views to {args.output}")


if __name__ == "__main__":
    main()
