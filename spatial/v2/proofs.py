"""Replayable proof certificates for SpatialEntail V2 problems.

This module is deliberately independent of the SMT implementation. It builds
and checks typed derivations from the structured premises themselves. The
automatic builder performs deterministic Boolean closure, bounded case splits,
axis reasoning, and formula refutation. The checker remains the independent
acceptance seam. Direction, Which, and Count answer sets are assembled by
certificate layers rather than solver-status fallbacks.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import (
    And,
    Direction,
    DirectionQuery,
    Iff,
    Implies,
    Not,
    Or,
    RelationConstraint,
    SpatialFormula,
    SpatialProblem,
    direction_signs,
)


class ProofConstructionError(ValueError):
    """The requested problem is outside the supported proof-first fragment."""


class ProofCheckError(ValueError):
    """A proof certificate contains an invalid or unsupported inference."""


class ProofRule(str, Enum):
    PREMISE = "premise"
    ASSUMPTION = "assumption"
    REFUTATION_ASSUMPTION = "refutation-assumption"
    AND_ELIMINATION = "and-elimination"
    AND_INTRODUCTION = "and-introduction"
    MODUS_PONENS = "modus-ponens"
    DISJUNCTIVE_SYLLOGISM = "disjunctive-syllogism"
    IFF_ELIMINATION = "iff-elimination"
    DOUBLE_NEGATION = "double-negation"
    CONTRADICTION = "contradiction"
    EXPLOSION = "explosion"
    CASE_SPLIT = "case-split"
    DIRECTION_DECOMPOSITION = "direction-decomposition"
    AXIS_INVERSION = "axis-inversion"
    AXIS_TRANSITIVITY = "axis-transitivity"
    AXIS_CONTRADICTION = "axis-contradiction"
    DIRECTION_RECOMPOSITION = "direction-recomposition"


class ProofAxis(str, Enum):
    X = "x"
    Y = "y"


class OrderRelation(str, Enum):
    LESS = "<"
    EQUAL = "="
    GREATER = ">"


@dataclass(frozen=True)
class AxisFact:
    axis: ProofAxis
    subject: str
    relation: OrderRelation
    reference: str


@dataclass(frozen=True)
class DirectionClaim:
    subject: str
    direction: Direction
    reference: str


@dataclass(frozen=True)
class Contradiction:
    """A branch contains both a formula and its explicit negation."""


ProofConclusion = SpatialFormula | AxisFact | DirectionClaim | Contradiction
_MAX_CASE_SPLIT_DEPTH = 4


@dataclass(frozen=True)
class ProofStep:
    id: str
    rule: ProofRule
    conclusion: ProofConclusion
    inputs: tuple[str, ...] = ()
    premise_index: int | None = None
    branch: str | None = None


@dataclass(frozen=True)
class DirectionProofCertificate:
    problem: SpatialProblem
    steps: tuple[ProofStep, ...]
    conclusion_step: str

    @property
    def conclusion(self) -> DirectionClaim:
        step = next(
            (item for item in self.steps if item.id == self.conclusion_step),
            None,
        )
        if step is None or not isinstance(step.conclusion, DirectionClaim):
            raise ProofCheckError("conclusion_step must identify a DirectionClaim")
        return step.conclusion

    @property
    def support_premise_indices(self) -> tuple[int, ...]:
        return _support_premise_indices(self.steps, self.conclusion_step)


@dataclass(frozen=True)
class DirectionRefutationCertificate:
    problem: SpatialProblem
    claim: RelationConstraint
    steps: tuple[ProofStep, ...]
    assumption_step: str
    contradiction_step: str

    @property
    def support_premise_indices(self) -> tuple[int, ...]:
        return _support_premise_indices(self.steps, self.contradiction_step)


@dataclass(frozen=True)
class FormulaRefutationCertificate:
    problem: SpatialProblem
    claim: SpatialFormula
    steps: tuple[ProofStep, ...]
    assumption_step: str
    contradiction_step: str

    @property
    def support_premise_indices(self) -> tuple[int, ...]:
        return _support_premise_indices(self.steps, self.contradiction_step)


def _relation_from_sign(sign: int) -> OrderRelation:
    return {
        -1: OrderRelation.LESS,
        0: OrderRelation.EQUAL,
        1: OrderRelation.GREATER,
    }[sign]


def _relation_sign(relation: OrderRelation) -> int:
    return {
        OrderRelation.LESS: -1,
        OrderRelation.EQUAL: 0,
        OrderRelation.GREATER: 1,
    }[relation]


def _inverse_relation(relation: OrderRelation) -> OrderRelation:
    return _relation_from_sign(-_relation_sign(relation))


def _axis_fact(atom: RelationConstraint, axis: ProofAxis) -> AxisFact:
    if len(atom.allowed) != 1:
        raise ProofConstructionError("proof premises must use exact directions")
    direction = next(iter(atom.allowed))
    axis_index = 0 if axis is ProofAxis.X else 1
    return AxisFact(
        axis,
        atom.subject,
        _relation_from_sign(direction_signs(direction)[axis_index]),
        atom.reference,
    )


def _inverse_fact(fact: AxisFact) -> AxisFact:
    return AxisFact(
        fact.axis,
        fact.reference,
        _inverse_relation(fact.relation),
        fact.subject,
    )


def _compose_relations(
    first: OrderRelation,
    second: OrderRelation,
) -> OrderRelation | None:
    first_sign = _relation_sign(first)
    second_sign = _relation_sign(second)
    if first_sign == 0:
        return second
    if second_sign == 0 or first_sign == second_sign:
        return first
    return None


def _compose_facts(first: AxisFact, second: AxisFact) -> AxisFact | None:
    if first.axis is not second.axis or first.reference != second.subject:
        return None
    relation = _compose_relations(first.relation, second.relation)
    if relation is None:
        return None
    return AxisFact(first.axis, first.subject, relation, second.reference)


def _reachable_step_ids(
    proof_steps: tuple[ProofStep, ...],
    conclusion_step: str,
) -> frozenset[str]:
    steps = {step.id: step for step in proof_steps}
    reachable: set[str] = set()
    frontier = [conclusion_step]
    while frontier:
        step_id = frontier.pop()
        if step_id in reachable:
            continue
        step = steps.get(step_id)
        if step is None:
            raise ProofCheckError(f"proof references unknown step: {step_id}")
        reachable.add(step_id)
        frontier.extend(step.inputs)
    return frozenset(reachable)


def _support_premise_indices(
    proof_steps: tuple[ProofStep, ...],
    conclusion_step: str,
) -> tuple[int, ...]:
    reachable = _reachable_step_ids(proof_steps, conclusion_step)
    return tuple(
        sorted(
            step.premise_index
            for step in proof_steps
            if step.id in reachable and step.premise_index is not None
        )
    )


def _top_level_premises(formula: SpatialFormula) -> tuple[SpatialFormula, ...]:
    return formula.operands if isinstance(formula, And) else (formula,)


def _positive_and_negative(
    first: SpatialFormula,
    second: SpatialFormula,
) -> tuple[SpatialFormula, Not] | None:
    if isinstance(first, Not) and first.operand == second:
        return second, first
    if isinstance(second, Not) and second.operand == first:
        return first, second
    return None


def _branch_parents(previous: Mapping[str, ProofStep]) -> dict[str, str | None]:
    parents: dict[str, str | None] = {}
    for step in previous.values():
        if step.branch is None:
            continue
        if step.rule is ProofRule.ASSUMPTION and step.inputs:
            source = previous.get(step.inputs[0])
            if source is not None:
                parents.setdefault(step.branch, source.branch)
        elif step.rule is ProofRule.REFUTATION_ASSUMPTION:
            parents.setdefault(step.branch, None)
    return parents


def _scope_is_accessible(
    source: str | None,
    target: str | None,
    parents: Mapping[str, str | None],
) -> bool:
    if source is None:
        return True
    cursor = target
    while cursor is not None:
        if cursor == source:
            return True
        cursor = parents.get(cursor)
    return False


def _check_step(
    step: ProofStep,
    previous: Mapping[str, ProofStep],
    premises: tuple[SpatialFormula, ...],
) -> None:
    inputs = tuple(previous.get(step_id) for step_id in step.inputs)
    if any(item is None for item in inputs):
        raise ProofCheckError(f"{step.id} depends on an unknown or later step")
    resolved_inputs = tuple(item for item in inputs if item is not None)

    if step.rule not in {ProofRule.CASE_SPLIT, ProofRule.ASSUMPTION}:
        parents = _branch_parents(previous)
        if any(
            not _scope_is_accessible(item.branch, step.branch, parents)
            for item in resolved_inputs
        ):
            message = (
                "leaks a branch result into global scope"
                if step.branch is None
                else "depends on another proof branch"
            )
            raise ProofCheckError(f"{step.id} {message}")

    if step.rule is ProofRule.PREMISE:
        if step.inputs or step.premise_index is None or step.branch is not None:
            raise ProofCheckError(f"{step.id} is not a valid premise step")
        if not 0 <= step.premise_index < len(premises):
            raise ProofCheckError(f"{step.id} has an invalid premise index")
        if step.conclusion != premises[step.premise_index]:
            raise ProofCheckError(f"{step.id} does not match its indexed premise")
        return

    if step.premise_index is not None:
        raise ProofCheckError(f"{step.id} assigns a premise index to a derived step")

    input_formulas = tuple(
        item.conclusion
        for item in resolved_inputs
        if isinstance(item.conclusion, SpatialFormula)
    )

    if step.rule is ProofRule.ASSUMPTION:
        if (
            step.branch is None
            or len(input_formulas) != 1
            or not isinstance(input_formulas[0], Or)
        ):
            raise ProofCheckError(f"{step.id} is not a scoped disjunct assumption")
        if step.conclusion not in input_formulas[0].operands:
            raise ProofCheckError(f"{step.id} assumes no disjunct from its source")
        if step.branch in _branch_parents(previous):
            raise ProofCheckError(f"{step.id} reuses an existing proof branch")
        return

    if step.rule is ProofRule.REFUTATION_ASSUMPTION:
        if (
            step.branch is None
            or step.inputs
            or not isinstance(step.conclusion, SpatialFormula)
        ):
            raise ProofCheckError(f"{step.id} is not a scoped refutation assumption")
        return

    if step.rule is ProofRule.AND_ELIMINATION:
        if len(input_formulas) != 1 or not isinstance(input_formulas[0], And):
            raise ProofCheckError(f"{step.id} must eliminate one conjunction")
        if step.conclusion not in input_formulas[0].operands:
            raise ProofCheckError(f"{step.id} concludes a non-conjunct")
        return

    if step.rule is ProofRule.AND_INTRODUCTION:
        if len(input_formulas) != len(resolved_inputs) or not input_formulas:
            raise ProofCheckError(f"{step.id} must combine formula inputs")
        if step.conclusion != And(input_formulas):
            raise ProofCheckError(f"{step.id} constructs the wrong conjunction")
        return

    if step.rule is ProofRule.MODUS_PONENS:
        if len(input_formulas) != 2:
            raise ProofCheckError(f"{step.id} requires an implication and antecedent")
        implication = next(
            (formula for formula in input_formulas if isinstance(formula, Implies)),
            None,
        )
        if implication is None:
            raise ProofCheckError(f"{step.id} has no implication input")
        antecedents = tuple(
            formula for formula in input_formulas if formula is not implication
        )
        if (
            antecedents != (implication.antecedent,)
            or step.conclusion != implication.consequent
        ):
            raise ProofCheckError(f"{step.id} is not a valid modus ponens step")
        return

    if step.rule is ProofRule.DISJUNCTIVE_SYLLOGISM:
        if len(input_formulas) != 2:
            raise ProofCheckError(f"{step.id} requires a disjunction and negation")
        disjunction = next(
            (formula for formula in input_formulas if isinstance(formula, Or)),
            None,
        )
        negation = next(
            (formula for formula in input_formulas if isinstance(formula, Not)),
            None,
        )
        if disjunction is None or negation is None:
            raise ProofCheckError(f"{step.id} lacks a disjunction or negation")
        if negation.operand not in disjunction.operands:
            raise ProofCheckError(f"{step.id} negates no disjunct")
        remaining = tuple(
            operand for operand in disjunction.operands if operand != negation.operand
        )
        expected: SpatialFormula = (
            remaining[0] if len(remaining) == 1 else Or(remaining)
        )
        if not remaining or step.conclusion != expected:
            raise ProofCheckError(f"{step.id} concludes the wrong remaining disjunct")
        return

    if step.rule is ProofRule.IFF_ELIMINATION:
        if len(input_formulas) != 2:
            raise ProofCheckError(f"{step.id} requires an equivalence and one side")
        equivalence = next(
            (formula for formula in input_formulas if isinstance(formula, Iff)),
            None,
        )
        if equivalence is None:
            raise ProofCheckError(f"{step.id} has no equivalence input")
        known = next(
            formula for formula in input_formulas if formula is not equivalence
        )
        if known == equivalence.left:
            expected = equivalence.right
        elif known == equivalence.right:
            expected = equivalence.left
        else:
            raise ProofCheckError(f"{step.id} proves neither side of the equivalence")
        if step.conclusion != expected:
            raise ProofCheckError(f"{step.id} concludes the wrong equivalent formula")
        return

    if step.rule is ProofRule.DOUBLE_NEGATION:
        if (
            len(input_formulas) != 1
            or not isinstance(input_formulas[0], Not)
            or not isinstance(input_formulas[0].operand, Not)
            or step.conclusion != input_formulas[0].operand.operand
        ):
            raise ProofCheckError(f"{step.id} is not a valid double-negation step")
        return

    if step.rule is ProofRule.CONTRADICTION:
        if (
            len(input_formulas) != 2
            or _positive_and_negative(*input_formulas) is None
            or not isinstance(step.conclusion, Contradiction)
        ):
            raise ProofCheckError(f"{step.id} is not an explicit contradiction")
        return

    if step.rule is ProofRule.EXPLOSION:
        if (
            len(resolved_inputs) != 1
            or not isinstance(resolved_inputs[0].conclusion, Contradiction)
            or not isinstance(step.conclusion, SpatialFormula)
            or step.branch is None
        ):
            raise ProofCheckError(f"{step.id} must derive a formula in a closed branch")
        return

    if step.rule is ProofRule.CASE_SPLIT:
        if len(resolved_inputs) < 3:
            raise ProofCheckError(f"{step.id} is not a complete case-split conclusion")
        source = resolved_inputs[0]
        if not isinstance(source.conclusion, Or) or source.branch != step.branch:
            raise ProofCheckError(f"{step.id} must split a disjunction in its scope")
        branch_results = resolved_inputs[1:]
        branches = tuple(item.branch for item in branch_results)
        if any(branch is None for branch in branches) or len(set(branches)) != len(
            branches
        ):
            raise ProofCheckError(f"{step.id} must use distinct scoped branch results")
        if not isinstance(step.conclusion, SpatialFormula) or any(
            item.conclusion != step.conclusion for item in branch_results
        ):
            raise ProofCheckError(f"{step.id} branches do not prove one conclusion")
        assumptions = tuple(
            item
            for item in previous.values()
            if item.rule is ProofRule.ASSUMPTION
            and item.branch in branches
            and item.inputs == (source.id,)
        )
        assumptions_by_branch: dict[str, ProofStep] = {}
        for assumption in assumptions:
            assert assumption.branch is not None
            if assumption.branch in assumptions_by_branch:
                raise ProofCheckError(
                    f"{step.id} has duplicate assumptions for one branch"
                )
            assumptions_by_branch[assumption.branch] = assumption
        parents = _branch_parents(previous)
        if (
            set(assumptions_by_branch) != set(branches)
            or any(parents.get(branch) != step.branch for branch in branches)
            or {item.conclusion for item in assumptions_by_branch.values()}
            != set(source.conclusion.operands)
        ):
            raise ProofCheckError(f"{step.id} does not cover every disjunct")
        return

    if step.rule is ProofRule.DIRECTION_DECOMPOSITION:
        if len(resolved_inputs) != 1 or not isinstance(
            resolved_inputs[0].conclusion, RelationConstraint
        ):
            raise ProofCheckError(f"{step.id} must decompose one spatial premise")
        if not isinstance(step.conclusion, AxisFact):
            raise ProofCheckError(f"{step.id} must conclude an axis fact")
        expected = _axis_fact(resolved_inputs[0].conclusion, step.conclusion.axis)
        if step.conclusion != expected:
            raise ProofCheckError(f"{step.id} has an invalid direction decomposition")
        return

    if step.rule is ProofRule.AXIS_INVERSION:
        if len(resolved_inputs) != 1 or not isinstance(
            resolved_inputs[0].conclusion, AxisFact
        ):
            raise ProofCheckError(f"{step.id} must invert one axis fact")
        if step.conclusion != _inverse_fact(resolved_inputs[0].conclusion):
            raise ProofCheckError(f"{step.id} has an invalid axis inversion")
        return

    if step.rule is ProofRule.AXIS_TRANSITIVITY:
        if len(resolved_inputs) != 2 or not all(
            isinstance(item.conclusion, AxisFact) for item in resolved_inputs
        ):
            raise ProofCheckError(f"{step.id} must compose two axis facts")
        expected = _compose_facts(
            resolved_inputs[0].conclusion,
            resolved_inputs[1].conclusion,
        )
        if expected is None or step.conclusion != expected:
            raise ProofCheckError(f"{step.id} has an invalid transitivity step")
        return

    if step.rule is ProofRule.AXIS_CONTRADICTION:
        if len(resolved_inputs) != 2 or not all(
            isinstance(item.conclusion, AxisFact) for item in resolved_inputs
        ):
            raise ProofCheckError(f"{step.id} must compare two axis facts")
        first = resolved_inputs[0].conclusion
        second = resolved_inputs[1].conclusion
        assert isinstance(first, AxisFact) and isinstance(second, AxisFact)
        if first.axis is not second.axis or {
            first.subject,
            first.reference,
        } != {second.subject, second.reference}:
            raise ProofCheckError(f"{step.id} compares different axis pairs")
        first_sign = _relation_sign(first.relation)
        second_sign = _relation_sign(second.relation)
        if first.subject == second.reference:
            second_sign = -second_sign
        if (
            first_sign == second_sign
            or not isinstance(step.conclusion, Contradiction)
            or step.branch is None
        ):
            raise ProofCheckError(f"{step.id} does not contain an axis contradiction")
        return

    if step.rule is ProofRule.DIRECTION_RECOMPOSITION:
        if len(resolved_inputs) != 2 or not all(
            isinstance(item.conclusion, AxisFact) for item in resolved_inputs
        ):
            raise ProofCheckError(f"{step.id} must combine two axis facts")
        if not isinstance(step.conclusion, DirectionClaim):
            raise ProofCheckError(f"{step.id} must conclude a direction claim")
        facts = {item.conclusion.axis: item.conclusion for item in resolved_inputs}
        if set(facts) != {ProofAxis.X, ProofAxis.Y}:
            raise ProofCheckError(f"{step.id} requires one X and one Y fact")
        x_fact = facts[ProofAxis.X]
        y_fact = facts[ProofAxis.Y]
        claim = step.conclusion
        actual_signs = []
        for fact in (x_fact, y_fact):
            sign = _relation_sign(fact.relation)
            if fact.subject == claim.subject and fact.reference == claim.reference:
                actual_signs.append(sign)
            elif fact.subject == claim.reference and fact.reference == claim.subject:
                actual_signs.append(-sign)
            else:
                raise ProofCheckError(f"{step.id} combines facts about different pairs")
        expected_signs = direction_signs(claim.direction)
        if tuple(actual_signs) != expected_signs:
            raise ProofCheckError(f"{step.id} recomposes the wrong direction")
        return

    raise ProofCheckError(f"unsupported proof rule: {step.rule}")


def _replay_steps(
    problem: SpatialProblem,
    steps: tuple[ProofStep, ...],
) -> dict[str, ProofStep]:
    premises = _top_level_premises(problem.premise)
    previous: dict[str, ProofStep] = {}
    for step in steps:
        if not step.id or step.id in previous:
            raise ProofCheckError("proof step identifiers must be non-empty and unique")
        _check_step(step, previous, premises)
        previous[step.id] = step
    return previous


def check_direction_proof(certificate: DirectionProofCertificate) -> None:
    """Replay a Direction certificate without consulting an SMT solver."""
    problem = certificate.problem
    query = problem.query
    if not isinstance(query, DirectionQuery):
        raise ProofCheckError("Direction certificates require a DirectionQuery")
    previous = _replay_steps(problem, certificate.steps)

    if certificate.conclusion_step not in previous:
        raise ProofCheckError("conclusion_step does not identify a proof step")
    conclusion = certificate.conclusion
    if (
        conclusion.subject != query.target
        or conclusion.reference != query.reference
        or conclusion.direction not in query.candidate_directions
    ):
        raise ProofCheckError("proof conclusion does not answer the DirectionQuery")
    _reachable_step_ids(certificate.steps, certificate.conclusion_step)


def _check_refutation(
    problem: SpatialProblem,
    claim: SpatialFormula,
    proof_steps: tuple[ProofStep, ...],
    assumption_step: str,
    contradiction_step: str,
) -> None:
    steps = _replay_steps(problem, proof_steps)
    assumption = steps.get(assumption_step)
    if (
        assumption is None
        or assumption.rule is not ProofRule.REFUTATION_ASSUMPTION
        or assumption.conclusion != claim
    ):
        raise ProofCheckError("assumption_step does not assume the refuted claim")
    contradiction = steps.get(contradiction_step)
    if contradiction is None or not isinstance(contradiction.conclusion, Contradiction):
        raise ProofCheckError("contradiction_step does not close the refutation")
    if contradiction.branch != assumption.branch:
        raise ProofCheckError(
            "refutation assumption and contradiction use different scopes"
        )
    reachable = _reachable_step_ids(proof_steps, contradiction_step)
    if assumption_step not in reachable:
        raise ProofCheckError("contradiction does not depend on the refuted claim")


def check_formula_refutation(certificate: FormulaRefutationCertificate) -> None:
    """Replay a contradiction for an arbitrary supported spatial formula."""
    try:
        SpatialProblem(
            certificate.problem.objects,
            And((certificate.problem.premise, certificate.claim)),
            certificate.problem.query,
        )
    except (TypeError, ValueError) as exc:
        raise ProofCheckError(f"invalid refutation claim: {exc}") from exc
    _check_refutation(
        certificate.problem,
        certificate.claim,
        certificate.steps,
        certificate.assumption_step,
        certificate.contradiction_step,
    )


def check_direction_refutation(
    certificate: DirectionRefutationCertificate,
) -> None:
    """Replay a contradiction showing that one Direction candidate is impossible."""
    problem = certificate.problem
    query = problem.query
    claim = certificate.claim
    if not isinstance(query, DirectionQuery):
        raise ProofCheckError("Direction refutations require a DirectionQuery")
    if (
        claim.subject != query.target
        or claim.reference != query.reference
        or len(claim.allowed) != 1
        or not claim.allowed <= query.candidate_directions
    ):
        raise ProofCheckError("refutation claim does not identify one query candidate")
    _check_refutation(
        problem,
        claim,
        certificate.steps,
        certificate.assumption_step,
        certificate.contradiction_step,
    )


@dataclass(frozen=True)
class _GraphEdge:
    destination: str
    strict: bool
    step_id: str
    invert: bool


_AxisPath = tuple[tuple[str, bool], ...]


def _proof_graph(
    objects: tuple[str, ...],
    steps: tuple[ProofStep, ...],
    axis: ProofAxis,
    branch: str | None = None,
) -> dict[str, list[_GraphEdge]]:
    adjacency = {obj: [] for obj in objects}
    for step in steps:
        if branch is None and step.branch is not None:
            continue
        if branch is not None and step.branch not in {None, branch}:
            continue
        fact = step.conclusion
        if not isinstance(fact, AxisFact) or fact.axis is not axis:
            continue
        if fact.relation is OrderRelation.LESS:
            adjacency[fact.subject].append(
                _GraphEdge(fact.reference, True, step.id, False)
            )
        elif fact.relation is OrderRelation.GREATER:
            adjacency[fact.reference].append(
                _GraphEdge(fact.subject, True, step.id, True)
            )
        elif fact.relation is OrderRelation.EQUAL:
            adjacency[fact.subject].append(
                _GraphEdge(fact.reference, False, step.id, False)
            )
            adjacency[fact.reference].append(
                _GraphEdge(fact.subject, False, step.id, True)
            )
    for edges in adjacency.values():
        edges.sort(key=lambda item: (item.destination, item.step_id))
    return adjacency


def _find_axis_path(
    adjacency: Mapping[str, list[_GraphEdge]],
    subject: str,
    reference: str,
    sign: int,
) -> _AxisPath | None:
    if sign > 0:
        start, end, require_strict = reference, subject, True
    elif sign < 0:
        start, end, require_strict = subject, reference, True
    else:
        start, end, require_strict = subject, reference, False

    initial = (start, False)
    frontier = deque([initial])
    visited = {initial}
    parents: dict[tuple[str, bool], tuple[tuple[str, bool], _GraphEdge]] = {}
    final: tuple[str, bool] | None = None
    while frontier:
        state = frontier.popleft()
        node, has_strict = state
        if node == end and (has_strict or not require_strict):
            final = state
            break
        for edge in adjacency[node]:
            if not require_strict and edge.strict:
                continue
            next_state = (edge.destination, has_strict or edge.strict)
            if next_state in visited:
                continue
            visited.add(next_state)
            parents[next_state] = (state, edge)
            frontier.append(next_state)
    if final is None:
        return None

    edge_steps: list[tuple[str, bool]] = []
    cursor = final
    while cursor != initial:
        previous, edge = parents[cursor]
        edge_steps.append((edge.step_id, edge.invert))
        cursor = previous
    edge_steps.reverse()
    return tuple(edge_steps)


def _derive_axis_path(
    axis: ProofAxis,
    edge_steps: _AxisPath,
    steps: list[ProofStep],
) -> str:
    if not edge_steps:
        raise ProofConstructionError("axis proof path cannot be empty")
    by_id = {step.id: step for step in steps}
    if len(edge_steps) == 1:
        return edge_steps[0][0]

    oriented_steps = []
    for index, (step_id, invert) in enumerate(edge_steps, start=1):
        if not invert:
            oriented_steps.append(step_id)
            continue
        fact = by_id[step_id].conclusion
        assert isinstance(fact, AxisFact)
        inverse_id = f"PATH-{axis.value.upper()}-{index}-INV"
        inverse = ProofStep(
            inverse_id,
            ProofRule.AXIS_INVERSION,
            _inverse_fact(fact),
            (step_id,),
        )
        steps.append(inverse)
        by_id[inverse_id] = inverse
        oriented_steps.append(inverse_id)

    current_id = oriented_steps[0]
    for index, next_id in enumerate(oriented_steps[1:], start=1):
        first = by_id[current_id].conclusion
        second = by_id[next_id].conclusion
        assert isinstance(first, AxisFact) and isinstance(second, AxisFact)
        conclusion = _compose_facts(first, second)
        if conclusion is None:
            raise ProofConstructionError("axis path is not transitively composable")
        current_id = f"D-{axis.value.upper()}-{index}"
        derived = ProofStep(
            current_id,
            ProofRule.AXIS_TRANSITIVITY,
            conclusion,
            (
                oriented_steps[0]
                if index == 1
                else f"D-{axis.value.upper()}-{index - 1}",
                next_id,
            ),
        )
        steps.append(derived)
        by_id[current_id] = derived
    return current_id


def proof_to_dict(certificate: DirectionProofCertificate) -> dict[str, Any]:
    """Return a deterministic JSON-compatible proof representation."""
    check_direction_proof(certificate)
    return tagged_dataclass_to_dict(certificate)


def refutation_to_dict(
    certificate: DirectionRefutationCertificate,
) -> dict[str, Any]:
    """Return a deterministic JSON-compatible refutation representation."""
    check_direction_refutation(certificate)
    return tagged_dataclass_to_dict(certificate)


def formula_refutation_to_dict(
    certificate: FormulaRefutationCertificate,
) -> dict[str, Any]:
    """Return a deterministic JSON-compatible formula-refutation representation."""
    check_formula_refutation(certificate)
    return tagged_dataclass_to_dict(certificate)


def _fresh_step_id(steps: list[ProofStep], prefix: str) -> str:
    existing = {step.id for step in steps}
    index = 1
    while f"{prefix}{index}" in existing:
        index += 1
    return f"{prefix}{index}"


def _premise_steps(problem: SpatialProblem) -> list[ProofStep]:
    return [
        ProofStep(
            f"P{index + 1}",
            ProofRule.PREMISE,
            premise,
            premise_index=index,
        )
        for index, premise in enumerate(_top_level_premises(problem.premise))
    ]


def _accessible_formula_steps(
    steps: list[ProofStep],
    branch: str | None,
) -> dict[SpatialFormula, str]:
    formulas: dict[SpatialFormula, str] = {}
    parents = _branch_parents({step.id: step for step in steps})
    for step in steps:
        if not isinstance(step.conclusion, SpatialFormula):
            continue
        if not _scope_is_accessible(step.branch, branch, parents):
            continue
        formulas[step.conclusion] = step.id
    return formulas


def _append_formula_step(
    steps: list[ProofStep],
    rule: ProofRule,
    conclusion: SpatialFormula,
    inputs: tuple[str, ...],
    branch: str | None,
) -> bool:
    if conclusion in _accessible_formula_steps(steps, branch):
        return False
    prefix = "F" if branch is None else f"{branch}-F"
    steps.append(
        ProofStep(
            _fresh_step_id(steps, prefix),
            rule,
            conclusion,
            inputs,
            branch=branch,
        )
    )
    return True


def _saturate_formula_steps(
    steps: list[ProofStep],
    branch: str | None = None,
) -> dict[SpatialFormula, str]:
    changed = True
    while changed:
        changed = False
        known = _accessible_formula_steps(steps, branch)
        snapshot = tuple(known.items())

        for formula, source_id in snapshot:
            if isinstance(formula, And):
                for operand in formula.operands:
                    changed |= _append_formula_step(
                        steps,
                        ProofRule.AND_ELIMINATION,
                        operand,
                        (source_id,),
                        branch,
                    )
            if isinstance(formula, Not) and isinstance(formula.operand, Not):
                changed |= _append_formula_step(
                    steps,
                    ProofRule.DOUBLE_NEGATION,
                    formula.operand.operand,
                    (source_id,),
                    branch,
                )

        known = _accessible_formula_steps(steps, branch)
        conjunctions = []
        for formula in known:
            candidates = (
                (formula.antecedent,)
                if isinstance(formula, Implies)
                else (formula.left, formula.right)
                if isinstance(formula, Iff)
                else ()
            )
            for candidate in candidates:
                if isinstance(candidate, And) and candidate not in conjunctions:
                    conjunctions.append(candidate)
        for conjunction in conjunctions:
            if all(operand in known for operand in conjunction.operands):
                changed |= _append_formula_step(
                    steps,
                    ProofRule.AND_INTRODUCTION,
                    conjunction,
                    tuple(known[operand] for operand in conjunction.operands),
                    branch,
                )

        known = _accessible_formula_steps(steps, branch)
        for formula, source_id in tuple(known.items()):
            if isinstance(formula, Implies) and formula.antecedent in known:
                changed |= _append_formula_step(
                    steps,
                    ProofRule.MODUS_PONENS,
                    formula.consequent,
                    (source_id, known[formula.antecedent]),
                    branch,
                )
            if isinstance(formula, Iff):
                for known_side, conclusion in (
                    (formula.left, formula.right),
                    (formula.right, formula.left),
                ):
                    if known_side in known:
                        changed |= _append_formula_step(
                            steps,
                            ProofRule.IFF_ELIMINATION,
                            conclusion,
                            (source_id, known[known_side]),
                            branch,
                        )
            if isinstance(formula, Or):
                for operand in formula.operands:
                    negation = Not(operand)
                    if negation not in known:
                        continue
                    remaining = tuple(
                        candidate
                        for candidate in formula.operands
                        if candidate != operand
                    )
                    if not remaining:
                        continue
                    conclusion: SpatialFormula = (
                        remaining[0] if len(remaining) == 1 else Or(remaining)
                    )
                    changed |= _append_formula_step(
                        steps,
                        ProofRule.DISJUNCTIVE_SYLLOGISM,
                        conclusion,
                        (source_id, known[negation]),
                        branch,
                    )
    return _accessible_formula_steps(steps, branch)


def _append_formula_contradiction(
    steps: list[ProofStep],
    branch: str,
    assumption_step: str,
) -> str | None:
    known = _accessible_formula_steps(steps, branch)
    for formula, formula_id in known.items():
        negation_id = known.get(Not(formula))
        if negation_id is None:
            continue
        if assumption_step not in (
            _reachable_step_ids(tuple(steps), formula_id)
            | _reachable_step_ids(tuple(steps), negation_id)
        ):
            continue
        contradiction_id = _fresh_step_id(steps, f"{branch}-C")
        steps.append(
            ProofStep(
                contradiction_id,
                ProofRule.CONTRADICTION,
                Contradiction(),
                (formula_id, negation_id),
                branch=branch,
            )
        )
        return contradiction_id
    return None


def _append_axis_decompositions(
    steps: list[ProofStep],
    branch: str | None = None,
) -> None:
    existing = {
        (step.inputs[0], step.conclusion.axis)
        for step in steps
        if step.rule is ProofRule.DIRECTION_DECOMPOSITION
        and step.inputs
        and isinstance(step.conclusion, AxisFact)
    }
    for source in tuple(steps):
        if not isinstance(source.conclusion, RelationConstraint):
            continue
        if len(source.conclusion.allowed) != 1:
            continue
        if branch is None and source.branch is not None:
            continue
        if branch is not None and source.branch not in {None, branch}:
            continue
        for axis in ProofAxis:
            if (source.id, axis) in existing:
                continue
            step_branch = source.branch
            prefix = (
                f"{source.id}-{axis.value.upper()}"
                if step_branch is None
                else f"{step_branch}-{axis.value.upper()}"
            )
            steps.append(
                ProofStep(
                    _fresh_step_id(steps, prefix),
                    ProofRule.DIRECTION_DECOMPOSITION,
                    _axis_fact(source.conclusion, axis),
                    (source.id,),
                    branch=step_branch,
                )
            )
            existing.add((source.id, axis))


def _derive_formula(
    steps: list[ProofStep],
    target: SpatialFormula,
    branch: str | None = None,
    depth: int = 0,
    max_depth: int = _MAX_CASE_SPLIT_DEPTH,
) -> str | None:
    known = _saturate_formula_steps(steps, branch)
    if target in known:
        return known[target]
    if isinstance(target, And):
        inputs = []
        for operand in target.operands:
            result = _derive_formula(steps, operand, branch, depth, max_depth)
            if result is None:
                break
            inputs.append(result)
        if len(inputs) == len(target.operands):
            _append_formula_step(
                steps,
                ProofRule.AND_INTRODUCTION,
                target,
                tuple(inputs),
                branch,
            )
            return _accessible_formula_steps(steps, branch).get(target)
    if depth >= max_depth:
        return None
    by_id = {step.id: step for step in steps}
    for source, source_id in tuple(known.items()):
        if (
            not isinstance(source, Or)
            or len(source.operands) < 2
            or by_id[source_id].branch != branch
        ):
            continue
        branch_results = []
        attempt_start = len(steps)
        for index, operand in enumerate(source.operands, start=1):
            child_branch = f"{branch + '/' if branch else ''}case-{source_id}-{index}"
            assumption_id = _fresh_step_id(steps, f"{child_branch}-A")
            steps.append(
                ProofStep(
                    assumption_id,
                    ProofRule.ASSUMPTION,
                    operand,
                    (source_id,),
                    branch=child_branch,
                )
            )
            branch_known = _saturate_formula_steps(steps, child_branch)
            result_id = branch_known.get(target)
            if result_id is None:
                contradiction_id = _append_formula_contradiction(
                    steps,
                    child_branch,
                    assumption_id,
                )
                if contradiction_id is not None:
                    result_id = _fresh_step_id(steps, f"{child_branch}-X")
                    steps.append(
                        ProofStep(
                            result_id,
                            ProofRule.EXPLOSION,
                            target,
                            (contradiction_id,),
                            branch=child_branch,
                        )
                    )
            if result_id is None:
                result_id = _derive_formula(
                    steps,
                    target,
                    child_branch,
                    depth + 1,
                    max_depth,
                )
            if result_id is None:
                del steps[attempt_start:]
                branch_results = []
                break
            branch_results.append(result_id)
        if branch_results:
            result_id = _fresh_step_id(steps, "CASE")
            steps.append(
                ProofStep(
                    result_id,
                    ProofRule.CASE_SPLIT,
                    target,
                    (source_id, *branch_results),
                    branch=branch,
                )
            )
            return result_id
    return None


def _direction_paths(
    problem: SpatialProblem,
    steps: list[ProofStep],
    candidates: list[Direction],
) -> dict[Direction, dict[ProofAxis, _AxisPath]]:
    _append_axis_decompositions(steps)
    graphs = {
        axis: _proof_graph(problem.objects, tuple(steps), axis) for axis in ProofAxis
    }
    query = problem.query
    assert isinstance(query, DirectionQuery)
    paths_by_direction = {}
    for candidate in candidates:
        x_sign, y_sign = direction_signs(candidate)
        paths = {
            ProofAxis.X: _find_axis_path(
                graphs[ProofAxis.X], query.target, query.reference, x_sign
            ),
            ProofAxis.Y: _find_axis_path(
                graphs[ProofAxis.Y], query.target, query.reference, y_sign
            ),
        }
        if all(path is not None for path in paths.values()):
            paths_by_direction[candidate] = {
                axis: path for axis, path in paths.items() if path is not None
            }
    return paths_by_direction


def build_direction_proof(
    problem: SpatialProblem,
    direction: Direction | None = None,
) -> DirectionProofCertificate:
    """Construct and replay a proof for one entailed exact Direction answer."""
    query = problem.query
    if not isinstance(query, DirectionQuery):
        raise ProofConstructionError(
            "proof construction currently supports DirectionQuery"
        )
    candidates = [direction] if direction is not None else list(Direction)
    candidates = [item for item in candidates if item in query.candidate_directions]
    steps = _premise_steps(problem)
    _saturate_formula_steps(steps)
    for candidate in candidates:
        _derive_formula(
            steps,
            RelationConstraint(
                query.target,
                query.reference,
                frozenset({candidate}),
            ),
        )
    paths_by_direction = _direction_paths(problem, steps, candidates)

    if direction is not None and direction not in paths_by_direction:
        raise ProofConstructionError(
            f"premises do not derive {direction.value} for the query pair"
        )
    if direction is None and len(paths_by_direction) != 1:
        names = ", ".join(item.value for item in paths_by_direction) or "none"
        raise ProofConstructionError(
            "proof construction requires one entailed direction; "
            f"derived candidates: {names}"
        )
    conclusion_direction = direction or next(iter(paths_by_direction))
    paths = paths_by_direction[conclusion_direction]
    x_step = _derive_axis_path(ProofAxis.X, paths[ProofAxis.X], steps)
    y_step = _derive_axis_path(ProofAxis.Y, paths[ProofAxis.Y], steps)
    conclusion_id = "Q-DIR"
    steps.append(
        ProofStep(
            conclusion_id,
            ProofRule.DIRECTION_RECOMPOSITION,
            DirectionClaim(
                query.target,
                conclusion_direction,
                query.reference,
            ),
            (x_step, y_step),
        )
    )
    reachable = _reachable_step_ids(tuple(steps), conclusion_id)
    certificate = DirectionProofCertificate(
        problem,
        tuple(step for step in steps if step.id in reachable),
        conclusion_id,
    )
    check_direction_proof(certificate)
    return certificate


def _axis_facts_conflict(first: AxisFact, second: AxisFact) -> bool:
    if first.axis is not second.axis or {
        first.subject,
        first.reference,
    } != {second.subject, second.reference}:
        return False
    first_sign = _relation_sign(first.relation)
    second_sign = _relation_sign(second.relation)
    if first.subject == second.reference:
        second_sign = -second_sign
    return first_sign != second_sign


def _append_axis_contradiction(
    problem: SpatialProblem,
    steps: list[ProofStep],
    branch: str,
    assumption_id: str,
) -> str | None:
    _append_axis_decompositions(steps, branch)
    global_graphs = {
        axis: _proof_graph(problem.objects, tuple(steps), axis) for axis in ProofAxis
    }
    branch_facts = [
        step
        for step in steps
        if isinstance(step.conclusion, AxisFact)
        and step.branch == branch
        and assumption_id in _reachable_step_ids(tuple(steps), step.id)
    ]
    for assumption_fact in branch_facts:
        fact = assumption_fact.conclusion
        assert isinstance(fact, AxisFact)
        desired_sign = _relation_sign(fact.relation)
        for conflicting_sign in (-1, 0, 1):
            if conflicting_sign == desired_sign:
                continue
            path = _find_axis_path(
                global_graphs[fact.axis],
                fact.subject,
                fact.reference,
                conflicting_sign,
            )
            if path is None:
                continue
            premise_fact = _derive_axis_path(fact.axis, path, steps)
            contradiction_id = _fresh_step_id(steps, f"{branch}-C")
            steps.append(
                ProofStep(
                    contradiction_id,
                    ProofRule.AXIS_CONTRADICTION,
                    Contradiction(),
                    (premise_fact, assumption_fact.id),
                    branch=branch,
                )
            )
            return contradiction_id

    accessible_facts = [
        step
        for step in steps
        if isinstance(step.conclusion, AxisFact) and step.branch in {None, branch}
    ]
    for index, first in enumerate(accessible_facts):
        for second in accessible_facts[index + 1 :]:
            if not _axis_facts_conflict(first.conclusion, second.conclusion):
                continue
            if assumption_id not in (
                _reachable_step_ids(tuple(steps), first.id)
                | _reachable_step_ids(tuple(steps), second.id)
            ):
                continue
            contradiction_id = _fresh_step_id(steps, f"{branch}-C")
            steps.append(
                ProofStep(
                    contradiction_id,
                    ProofRule.AXIS_CONTRADICTION,
                    Contradiction(),
                    (first.id, second.id),
                    branch=branch,
                )
            )
            return contradiction_id
    return None


def build_formula_refutation(
    problem: SpatialProblem,
    claim: SpatialFormula,
) -> FormulaRefutationCertificate:
    """Construct and replay a contradiction for a supported Boolean claim."""
    steps = _premise_steps(problem)
    _saturate_formula_steps(steps)
    if isinstance(problem.query, DirectionQuery):
        for direction in Direction:
            if direction not in problem.query.candidate_directions:
                continue
            _derive_formula(
                steps,
                RelationConstraint(
                    problem.query.target,
                    problem.query.reference,
                    frozenset({direction}),
                ),
            )
    branch = "refutation"
    assumption_id = "A-REFUTE"
    steps.append(
        ProofStep(
            assumption_id,
            ProofRule.REFUTATION_ASSUMPTION,
            claim,
            branch=branch,
        )
    )
    _saturate_formula_steps(steps, branch)
    contradiction_id = _append_formula_contradiction(steps, branch, assumption_id)
    if contradiction_id is None:
        contradiction_id = _append_axis_contradiction(
            problem,
            steps,
            branch,
            assumption_id,
        )
    if contradiction_id is None:
        raise ProofConstructionError("premises do not refute the supplied formula")
    reachable = _reachable_step_ids(tuple(steps), contradiction_id)
    certificate = FormulaRefutationCertificate(
        problem,
        claim,
        tuple(step for step in steps if step.id in reachable),
        assumption_id,
        contradiction_id,
    )
    check_formula_refutation(certificate)
    return certificate


def build_direction_refutation(
    problem: SpatialProblem,
    direction: Direction,
) -> DirectionRefutationCertificate:
    """Construct an axis or Boolean contradiction for one Direction candidate."""
    query = problem.query
    if not isinstance(query, DirectionQuery):
        raise ProofConstructionError(
            "refutation construction currently supports DirectionQuery"
        )
    if direction not in query.candidate_directions:
        raise ProofConstructionError(
            "refuted direction is outside the query candidates"
        )
    claim = RelationConstraint(
        query.target,
        query.reference,
        frozenset({direction}),
    )
    try:
        formula_refutation = build_formula_refutation(problem, claim)
    except ProofConstructionError as exc:
        raise ProofConstructionError(
            f"premises do not refute {direction.value} for the query pair"
        ) from exc
    certificate = DirectionRefutationCertificate(
        problem,
        claim,
        formula_refutation.steps,
        formula_refutation.assumption_step,
        formula_refutation.contradiction_step,
    )
    check_direction_refutation(certificate)
    return certificate
