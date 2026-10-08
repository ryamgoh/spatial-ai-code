"""Oracle-assisted construction of independently checked answer certificates."""

from __future__ import annotations

from typing import TypeAlias

from spatial.v2.answer_certificates import (
    DirectionAnswerSetCertificate,
    answer_set_to_dict,
    build_direction_answer_set,
)
from spatial.v2.count_certificates import (
    CountAnswerSetCertificate,
    CountMembershipConflict,
    build_count_answer_set,
    count_answer_set_to_dict,
    count_assignment_formula,
    count_assignments,
)
from spatial.v2.proofs import ProofConstructionError, build_formula_refutation
from spatial.v2.solver import (
    CountAnalysis,
    CountQuery,
    DirectionAnalysis,
    DirectionQuery,
    QueryAnalysis,
    SpatialProblem,
    SpatialSolverV2,
    WhichAnalysis,
    WhichQuery,
    membership_constraint,
)
from spatial.v2.which_certificates import (
    MembershipStatus,
    WhichAnswerSetCertificate,
    build_which_answer_set,
    which_answer_set_to_dict,
)

AnswerCertificate: TypeAlias = (
    DirectionAnswerSetCertificate
    | WhichAnswerSetCertificate
    | CountAnswerSetCertificate
)


def _membership_models(
    problem: SpatialProblem,
    query: WhichQuery,
    solver: SpatialSolverV2,
) -> tuple[
    dict[str, dict[str, tuple[int, int]]],
    dict[str, dict[str, tuple[int, int]]],
]:
    positive = {}
    negative = {}
    for candidate in query.candidates:
        claim = membership_constraint(candidate, query)
        assessment = solver.assess(problem, claim)
        if assessment.error or not assessment.consistent:
            raise ProofConstructionError(
                assessment.error or f"cannot assess membership for {candidate}"
            )
        if assessment.witness is not None:
            positive[candidate] = assessment.witness
        if assessment.counterexample is not None:
            negative[candidate] = assessment.counterexample
    return positive, negative


def _build_which_certificate(
    problem: SpatialProblem,
    query: WhichQuery,
    solver: SpatialSolverV2,
) -> WhichAnswerSetCertificate:
    positive, negative = _membership_models(problem, query, solver)
    return build_which_answer_set(
        problem,
        positive_models=positive,
        negative_models=negative,
    )


def _build_count_certificate(
    problem: SpatialProblem,
    query: CountQuery,
    analysis: CountAnalysis,
    solver: SpatialSolverV2,
) -> CountAnswerSetCertificate:
    which_query = WhichQuery(query.directions, query.reference, query.candidates)
    which_problem = SpatialProblem(problem.objects, problem.premise, which_query)
    memberships = _build_which_certificate(which_problem, which_query, solver)
    possible = set(analysis.possible_counts)
    impossible_refutations = {}
    for count in range(len(query.candidates) + 1):
        if count in possible:
            continue
        conflicts = []
        for members in count_assignments(query, count):
            conflict = next(
                (
                    candidate
                    for candidate in memberships.candidates
                    if (
                        candidate.candidate in members
                        and candidate.status is MembershipStatus.IMPOSSIBLE
                    )
                    or (
                        candidate.candidate not in members
                        and candidate.status is MembershipStatus.ENTAILED
                    )
                ),
                None,
            )
            if conflict is None:
                conflicts.append(
                    build_formula_refutation(
                        problem,
                        count_assignment_formula(query, members),
                    )
                )
            else:
                conflicts.append(
                    CountMembershipConflict(conflict.candidate, conflict.evidence)
                )
        impossible_refutations[count] = tuple(conflicts)
    return build_count_answer_set(
        problem,
        possible_models=analysis.witnesses,
        impossible_refutations=impossible_refutations,
    )


def build_answer_certificate(
    problem: SpatialProblem,
    analysis: QueryAnalysis,
    solver: SpatialSolverV2,
) -> AnswerCertificate:
    """Turn oracle models into a certificate replayable without the oracle."""
    query = problem.query
    if isinstance(query, DirectionQuery) and isinstance(analysis, DirectionAnalysis):
        return build_direction_answer_set(problem, analysis.witnesses)
    if isinstance(query, WhichQuery) and isinstance(analysis, WhichAnalysis):
        return _build_which_certificate(problem, query, solver)
    if isinstance(query, CountQuery) and isinstance(analysis, CountAnalysis):
        return _build_count_certificate(problem, query, analysis, solver)
    raise TypeError("problem query and semantic analysis types do not match")


def answer_certificate_to_dict(certificate: AnswerCertificate) -> dict[str, object]:
    if isinstance(certificate, DirectionAnswerSetCertificate):
        return answer_set_to_dict(certificate)
    if isinstance(certificate, WhichAnswerSetCertificate):
        return which_answer_set_to_dict(certificate)
    return count_answer_set_to_dict(certificate)
