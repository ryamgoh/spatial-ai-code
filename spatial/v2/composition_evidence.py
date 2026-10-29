"""Matched composition evidence beyond unique Direction, with bounded support.

Reuses shared Direction/Selection certificates. Count exclusions are justified
by fixed memberships and checked pair incompatibilities, not by adding marginal
possibilities. Higher-order correlations outside this fragment are rejected.
"""

from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import combinations

from spatial.v2.answer_certificates import (
    DirectionAnswerSetCertificate,
    DirectionEntailmentCertificate,
    check_direction_answer_set,
)
from spatial.v2.certificate_generation import build_answer_certificate
from spatial.v2.composition_conflicts import (
    Conflict,
    build_conflict,
    render_conflict,
    verify_conflict,
)
from spatial.v2.composition_contracts import (
    VERSION,
    Semantics,
    UnsupportedEvidence,
    VerificationError,
    analyze,
    answer_domain,
    check_coordinates,
    check_witness,
    encode,
    parse_answer,
    parse_prompt,
    render_prompt,
    witness_outcome,
)
from spatial.v2.composition_direction import _fact_text
from spatial.v2.grading import AnswerMode
from spatial.v2.model_certificates import (
    ContingencyCertificate,
    SpatialModelCertificate,
)
from spatial.v2.proofs import (
    Contradiction,
    DirectionProofCertificate,
    DirectionRefutationCertificate,
    ProofConstructionError,
    ProofRule,
)
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import (
    And,
    CountQuery,
    DirectionQuery,
    Not,
    RelationConstraint,
    SpatialProblem,
    SpatialSolverV2,
    WhichQuery,
    membership_constraint,
)
from spatial.v2.which_certificates import (
    MembershipEntailmentCertificate,
    MembershipImpossibilityCertificate,
    MembershipProofCertificate,
    WhichAnswerSetCertificate,
    check_which_answer_set,
)


@dataclass(frozen=True)
class IncompatiblePair:
    members: tuple[str, str]
    conflict: Conflict


@dataclass(frozen=True)
class CountSupport:
    problem: SpatialProblem
    memberships: WhichAnswerSetCertificate
    incompatible_pairs: tuple[IncompatiblePair, ...]


@dataclass(frozen=True)
class Evidence:
    semantics: Semantics
    certificate: (
        DirectionAnswerSetCertificate
        | WhichAnswerSetCertificate
        | CountSupport
        | Conflict
    )


def _walk(value):
    yield value
    if is_dataclass(value):
        for field in fields(value):
            yield from _walk(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _walk(item)


def _checked_analysis(problem, solver):
    try:
        result = solver.analyze(problem, include_witnesses=True)
    except RuntimeError as exc:
        raise VerificationError(str(exc)) from exc
    if result.error is not None or not result.consistent:
        raise VerificationError(
            result.error or "unexpected inconsistent certificate input"
        )
    return result


def _certificate(problem, solver):
    try:
        return build_answer_certificate(
            problem, _checked_analysis(problem, solver), solver
        )
    except ProofConstructionError as exc:
        raise UnsupportedEvidence(
            f"shared certificate construction failed: {exc}"
        ) from exc


def _which_projection(problem):
    query = problem.query
    return replace(
        problem, query=WhichQuery(query.directions, query.reference, query.candidates)
    )


def _pair_problem(problem, members):
    return replace(
        problem,
        premise=And(
            (
                *problem.premise.operands,
                *(membership_constraint(obj, problem.query) for obj in members),
            )
        ),
    )


def _count_support(problem, solver):
    if len(problem.query.candidates) > 8:
        raise UnsupportedEvidence(
            "Count evidence currently supports at most eight eligible objects"
        )
    memberships = _certificate(_which_projection(problem), solver)
    pairs = []
    for members in combinations(memberships.possible_entities, 2):
        try:
            conflict = build_conflict(_pair_problem(problem, members))
        except UnsupportedEvidence:
            continue
        pairs.append(IncompatiblePair(members, conflict))
    return CountSupport(problem, memberships, tuple(pairs))


def _check_count(support: CountSupport, possible, solver):
    problem, memberships = support.problem, support.memberships
    if not isinstance(problem.query, CountQuery) or len(problem.query.candidates) > 8:
        raise UnsupportedEvidence("unsupported Count evidence domain")
    if memberships.problem != _which_projection(problem):
        raise VerificationError("Count membership evidence uses the wrong problem")
    check_which_answer_set(memberships)
    seen = set()
    for pair in support.incompatible_pairs:
        if (
            len(pair.members) != 2
            or len(set(pair.members)) != 2
            or not set(pair.members) <= set(problem.query.candidates)
            or pair.members in seen
            or pair.conflict.problem != _pair_problem(problem, pair.members)
        ):
            raise VerificationError("invalid Count pair conflict")
        seen.add(pair.members)
        verify_conflict(pair.conflict, solver)
    required = set(memberships.entailed_entities)
    excluded = set(memberships.impossible_entities)
    for count in answer_domain(problem):
        if count in possible:
            continue
        for assignment in combinations(problem.query.candidates, count):
            members = set(assignment)
            if not required <= members or excluded & members:
                continue
            if any(set(pair.members) <= members for pair in support.incompatible_pairs):
                continue
            raise UnsupportedEvidence(
                f"count {count} lacks a checked exclusion for joint assignment {assignment}"
            )


def build_evidence(problem: SpatialProblem, *, solver=None) -> Evidence:
    solver = solver or SpatialSolverV2(backend="z3")
    semantics = analyze(problem, solver=solver)
    if not semantics.consistent:
        certificate = build_conflict(problem)
    elif isinstance(problem.query, CountQuery):
        certificate = _count_support(problem, solver)
    else:
        certificate = _certificate(problem, solver)
    evidence = Evidence(semantics, certificate)
    verify_evidence(evidence, solver=solver)
    return evidence


def verify_evidence(evidence: Evidence, *, solver=None) -> None:
    solver = solver or SpatialSolverV2(backend="z3")
    semantics, certificate = evidence.semantics, evidence.certificate
    problem = semantics.problem
    if certificate.problem != problem:
        raise VerificationError("certificate and semantics refer to different problems")
    actual = analyze(problem, solver=solver)
    if (actual.consistent, actual.possible, actual.certain) != (
        semantics.consistent,
        semantics.possible,
        semantics.certain,
    ):
        raise VerificationError("evidence disagrees with full-domain solver analysis")
    if set(semantics.witnesses) != semantics.possible:
        raise VerificationError("stored witnesses must cover every possible value")
    for value, coordinates in semantics.witnesses.items():
        check_witness(problem, coordinates, value)
    if isinstance(certificate, Conflict):
        if actual.consistent:
            raise VerificationError("conflict supplied for a consistent map")
        verify_conflict(certificate, solver)
    elif not actual.consistent:
        raise VerificationError("inconsistent maps require conflict evidence")
    elif isinstance(certificate, DirectionAnswerSetCertificate):
        check_direction_answer_set(certificate)
        if (
            frozenset(d.value for d in certificate.possible_directions)
            != actual.possible
        ):
            raise VerificationError("Direction certificate possibility mismatch")
    elif isinstance(certificate, WhichAnswerSetCertificate):
        check_which_answer_set(certificate)
        if (
            frozenset(certificate.possible_entities) != actual.possible
            or frozenset(certificate.entailed_entities) != actual.certain
        ):
            raise VerificationError("Selection certificate membership mismatch")
    elif isinstance(certificate, CountSupport):
        _check_count(certificate, actual.possible, solver)
    else:
        raise UnsupportedEvidence("unknown evidence certificate type")
    for item in _walk(certificate):
        if isinstance(item, SpatialModelCertificate):
            # The shared checker evaluates claims; this separately checks all
            # premises and distinctness without its direction helper.
            check_coordinates(item.problem, item.coordinates)
            if isinstance(item.claim, RelationConstraint):
                projection = replace(
                    item.problem,
                    query=WhichQuery(
                        item.claim.allowed, item.claim.reference, (item.claim.subject,)
                    ),
                )
                outcome = bool(witness_outcome(projection, item.coordinates))
                if outcome != item.expected_claim_value:
                    raise VerificationError(
                        "independent model evaluation contradicts the claim"
                    )
        if isinstance(
            item, (DirectionProofCertificate, DirectionRefutationCertificate)
        ):
            _render_proof(item, symbolic=False)
            _render_proof(item, symbolic=True)


def _formula_text(fact, symbolic):
    if isinstance(fact, Not):
        body = _formula_text(fact.operand, symbolic)
        return f"NOT({body})" if symbolic else f"it is not the case that {body}"
    if isinstance(fact, And):
        return (" AND " if symbolic else "; and ").join(
            _formula_text(part, symbolic) for part in fact.operands
        )
    if isinstance(fact, Contradiction):
        return "contradiction"
    return _fact_text(fact, symbolic=symbolic)


def _render_proof(proof, *, symbolic):
    allowed = {
        ProofRule.PREMISE,
        ProofRule.DIRECTION_DECOMPOSITION,
        ProofRule.AXIS_INVERSION,
        ProofRule.AXIS_TRANSITIVITY,
        ProofRule.DIRECTION_INTRODUCTION,
        ProofRule.DIRECTION_RECOMPOSITION,
        ProofRule.ATOM_RECOMPOSITION,
        ProofRule.REFUTATION_ASSUMPTION,
        ProofRule.AXIS_CONTRADICTION,
        ProofRule.CONTRADICTION,
        ProofRule.AND_ELIMINATION,
    }
    lines = []
    for step in proof.steps:
        if step.rule not in allowed:
            raise UnsupportedEvidence(
                f"unimplemented matched rendering for {step.rule.value}"
            )
        fact = _formula_text(step.conclusion, symbolic)
        if step.rule is ProofRule.REFUTATION_ASSUMPTION:
            lines.append(f"{step.id}: Suppose {fact} for contradiction.")
            continue
        source = (
            f"p{step.premise_index + 1}"
            if step.rule is ProofRule.PREMISE
            else ", ".join(step.inputs)
        )
        scope = f"under assumption {step.branch}; " if step.branch else ""
        lines.append(
            f"{step.id}: [{scope}{source}] => {fact}"
            if symbolic
            else f"{step.id}: From {scope}{source}, {fact}."
        )
    if isinstance(proof, DirectionRefutationCertificate):
        claim = _formula_text(proof.claim, symbolic)
        lines.append(
            f"The assumption {claim} leads to contradiction, so this direction is impossible."
        )
    return "\n".join(lines)


def _model_text(problem, coordinates, *, symbolic):
    outcome = witness_outcome(problem, coordinates)
    if isinstance(outcome, frozenset):
        result = "qualifying objects={" + ", ".join(sorted(outcome)) + "}"
    else:
        result = f"{'direction' if isinstance(problem.query, DirectionQuery) else 'count'}={outcome}"
    if symbolic:
        points = ", ".join(f"{obj}:({x},{y})" for obj, (x, y) in coordinates.items())
        return f"Valid map {{{points}}} => {result}."
    points = "; ".join(f"{obj} at ({x}, {y})" for obj, (x, y) in coordinates.items())
    return f"A valid map places {points}. In this map, {result}."


def _membership_text(item, *, symbolic):
    evidence = item.evidence
    if isinstance(evidence, MembershipProofCertificate):
        return _render_proof(evidence.proof, symbolic=symbolic)
    if isinstance(
        evidence, (MembershipEntailmentCertificate, MembershipImpossibilityCertificate)
    ):
        return "\n".join(
            _render_proof(proof, symbolic=symbolic) for proof in evidence.refutations
        )
    if isinstance(evidence, ContingencyCertificate):
        return (
            "\n".join(
                _model_text(model.problem, model.coordinates, symbolic=symbolic)
                for model in (evidence.witness, evidence.counterexample)
            )
            + f"\n{item.candidate} qualifies in the first map but not the second."
        )
    raise UnsupportedEvidence("unknown membership rendering")


def _direction_text(certificate, values, *, symbolic):
    lines = []
    for item in certificate.candidates:
        if item.direction.value not in values:
            continue
        evidence = item.evidence
        if isinstance(evidence, DirectionEntailmentCertificate):
            text = _render_proof(evidence.proof, symbolic=symbolic)
        elif isinstance(evidence, SpatialModelCertificate):
            text = _model_text(
                evidence.problem, evidence.coordinates, symbolic=symbolic
            )
        else:
            text = _render_proof(evidence, symbolic=symbolic)
        lines.append(f"{item.direction.value}:\n{text}")
    return "\n".join(lines)


def _count_text(evidence, values, *, symbolic):
    support = evidence.certificate
    lines = []
    # Fixed memberships and pair conflicts cover every excluded count jointly.
    if set(values) - evidence.semantics.possible:
        for item in support.memberships.candidates:
            if item.candidate in support.memberships.contingent_entities:
                continue
            lines.append(_membership_text(item, symbolic=symbolic))
            lines.append(f"Fixed membership: {item.candidate} is {item.status.value}.")
        for pair in support.incompatible_pairs:
            first_index = len(support.problem.premise.operands) + 1
            assumptions = "; ".join(
                f"p{first_index + i}: {_formula_text(membership_constraint(obj, support.problem.query), symbolic)}"
                for i, obj in enumerate(pair.members)
            )
            lines.append(f"Suppose both qualify: {assumptions}.")
            lines.append(render_conflict(pair.conflict, symbolic=symbolic))
            lines.append(
                f"Thus {pair.members[0]} and {pair.members[1]} cannot qualify together."
            )
        lines.append(
            "Every joint membership assignment must include all fixed matches, "
            "exclude all fixed nonmatches, and contain no incompatible pair."
        )
    for value in values:
        if value in evidence.semantics.possible:
            lines.append(
                _model_text(
                    support.problem,
                    evidence.semantics.witnesses[value],
                    symbolic=symbolic,
                )
            )
        else:
            lines.append(
                f"Count {value} is impossible: every eligible-object subset of that "
                "size violates at least one of the established membership constraints."
            )
    return "\n".join(lines)


def render_pair(
    evidence: Evidence,
    menu,
    mode: AnswerMode,
    *,
    solver=None,
    compact_unique_direction: bool = False,
) -> dict[str, str]:
    mode = AnswerMode(mode)
    verify_evidence(evidence, solver=solver)
    semantics, certificate = evidence.semantics, evidence.certificate
    decision = encode(semantics, menu, mode)
    domain = answer_domain(semantics.problem)
    possible = semantics.possible
    if decision.escape_reason == "incomplete-coverage":
        selected = (
            next(value for value in domain if value in possible and value not in menu),
        )
        conclusion = f"The possible value {selected[0]} is unlisted, so A-D cannot form a complete answer."
        selection = "unlisted-witness"
    elif not semantics.consistent:
        selected, selection = (), "conflict"
        conclusion = "No valid map exists; select E regardless of the query."
    elif (
        mode is AnswerMode.SINGLE
        and semantics.case == "underdetermined"
        and not isinstance(semantics.problem.query, WhichQuery)
    ):
        selected = tuple(value for value in domain if value in possible)[:2]
        selection = "different-outcomes"
        conclusion = (
            "These valid maps give different answers, so there is no invariant answer."
        )
    else:
        selected = tuple(menu) if mode is AnswerMode.VISIBLE_POSSIBLE else domain
        selection = (
            "listed-domain" if mode is AnswerMode.VISIBLE_POSSIBLE else "full-domain"
        )
        if mode is AnswerMode.SINGLE:
            conclusion = (
                {
                    "unique": (
                        f"The uniquely guaranteed answer is {next(iter(possible))}. "
                        + (
                            "It is absent from A-D."
                            if decision.letters == "F"
                            else f"It is displayed at {decision.letters}."
                        )
                    ),
                    "no-match": "No eligible object can qualify, so there is no unique qualifying object.",
                    "multiple-certain-match": "Multiple objects always qualify, so there is no sole qualifying object.",
                    "underdetermined": "The memberships do not establish one object as the only match in every valid map.",
                }[semantics.case]
                if possible
                else "No eligible object can qualify, so there is no unique qualifying object."
            )
        else:
            values = tuple(value for value in selected if value in possible)
            conclusion = (
                f"The {'listed' if mode is AnswerMode.VISIBLE_POSSIBLE else 'complete'} possible set is "
                "{" + ", ".join(map(str, values)) + "}."
            )
            if not values:
                conclusion += " The empty set uses F."
    if (
        compact_unique_direction
        and mode is AnswerMode.SINGLE
        and semantics.case == "unique"
        and isinstance(certificate, DirectionAnswerSetCertificate)
    ):
        entailed = [
            item.direction.value
            for item in certificate.candidates
            if isinstance(item.evidence, DirectionEntailmentCertificate)
            and item.direction.value in possible
        ]
        if len(entailed) == 1:
            selected = tuple(entailed)
    pair = {}
    for style in ("natural", "symbolic"):
        symbolic = style == "symbolic"
        if selection == "conflict":
            body = render_conflict(certificate, symbolic=symbolic)
        elif selection == "unlisted-witness":
            body = _model_text(
                semantics.problem, semantics.witnesses[selected[0]], symbolic=symbolic
            )
        elif isinstance(certificate, DirectionAnswerSetCertificate):
            body = _direction_text(certificate, selected, symbolic=symbolic)
        elif isinstance(certificate, WhichAnswerSetCertificate):
            body = "\n".join(
                f"{item.candidate}: {item.status.value}\n{_membership_text(item, symbolic=symbolic)}"
                for item in certificate.candidates
                if item.candidate in selected
            )
        else:
            body = _count_text(evidence, selected, symbolic=symbolic)
        pair[style] = body + f"\n{conclusion}\nAnswer: {decision.letters}"
        if parse_answer(pair[style], mode, stage=1) != decision.letters:
            raise VerificationError("invalid rendered answer")
    return pair


def build_example(problem, menu, mode, *, story_id, family_id):
    if not story_id or not family_id:
        raise ValueError("story_id and family_id are required")
    mode = AnswerMode(mode)
    evidence = build_evidence(problem)
    prompt = render_prompt(problem, menu, mode)
    if parse_prompt(prompt) != (problem, menu, mode):
        raise VerificationError("full prompt round trip changed the problem")
    pair = render_pair(evidence, menu, mode)
    decision = encode(evidence.semantics, menu, mode)
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
            "evidence": tagged_dataclass_to_dict(evidence),
            "semantic_case": decision.semantic_case,
            "escape_reason": decision.escape_reason,
        },
    }
