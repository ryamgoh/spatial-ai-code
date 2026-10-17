"""Certificate-derived structural measurements for SpatialEntail problems.

The generator may request a particular structure, but only the replayed proof
graph establishes the realised semantic dependency length.  This module keeps
that measurement independent from generation intent and from model scoring.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Iterable

from spatial.v2.proofs import (
    DirectionProofCertificate,
    DirectionRefutationCertificate,
    FormulaRefutationCertificate,
    ProofRule,
    ProofStep,
)
from spatial.v2.solver import (
    And,
    Iff,
    Implies,
    Not,
    Or,
    RelationConstraint,
    SpatialFormula,
    SpatialProblem,
)


SEMANTIC_RULES = frozenset(
    {
        ProofRule.MODUS_PONENS,
        ProofRule.DISJUNCTIVE_SYLLOGISM,
        ProofRule.IFF_ELIMINATION,
        ProofRule.DOUBLE_NEGATION,
        ProofRule.CASE_SPLIT,
        ProofRule.AXIS_TRANSITIVITY,
    }
)

_NON_INFERENTIAL_RULES = frozenset(
    {
        ProofRule.PREMISE,
        ProofRule.ASSUMPTION,
        ProofRule.REFUTATION_ASSUMPTION,
    }
)


@dataclass(frozen=True)
class ProofGraphMeasurement:
    """Measurements for one positive proof or refutation DAG."""

    semantic_length: int
    full_depth: int
    semantic_size: int
    branch_steps: int
    merge_steps: int
    support_premise_indices: tuple[int, ...]
    operator_counts: dict[str, int]
    critical_signatures: tuple[tuple[str, ...], ...]
    ordered_transitions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ProofComplexityProfile:
    """Problem-level input measurements and certificate-wide proof aggregates."""

    top_level_premises: int
    atomic_occurrences: int
    unique_atoms: int
    connective_counts: dict[str, int]
    formula_nesting: int
    semantic_length: int
    full_certificate_depth: int
    semantic_proof_size: int
    branch_steps: int
    merge_steps: int
    proof_obligations: int
    support_premise_indices: tuple[int, ...]
    supporting_premises: int
    removable_premises: int
    operator_counts: dict[str, int]
    critical_signatures: tuple[tuple[str, ...], ...]
    ordered_transitions: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _formula_measurements(
    formula: SpatialFormula,
) -> tuple[list[RelationConstraint], Counter[str], int]:
    if isinstance(formula, RelationConstraint):
        return [formula], Counter(), 1
    if isinstance(formula, Not):
        atoms, counts, depth = _formula_measurements(formula.operand)
        counts["not"] += 1
        return atoms, counts, depth + 1
    if isinstance(formula, (And, Or)):
        atoms: list[RelationConstraint] = []
        counts: Counter[str] = Counter()
        depths = []
        for operand in formula.operands:
            child_atoms, child_counts, child_depth = _formula_measurements(operand)
            atoms.extend(child_atoms)
            counts.update(child_counts)
            depths.append(child_depth)
        counts["and" if isinstance(formula, And) else "or"] += 1
        return atoms, counts, 1 + max(depths, default=0)
    if isinstance(formula, Implies):
        left = _formula_measurements(formula.antecedent)
        right = _formula_measurements(formula.consequent)
        counts = left[1] + right[1]
        counts["implies"] += 1
        return left[0] + right[0], counts, 1 + max(left[2], right[2])
    if isinstance(formula, Iff):
        left = _formula_measurements(formula.left)
        right = _formula_measurements(formula.right)
        counts = left[1] + right[1]
        counts["iff"] += 1
        return left[0] + right[0], counts, 1 + max(left[2], right[2])
    raise TypeError(f"unsupported spatial formula: {type(formula).__name__}")


def _reachable_steps(
    steps: tuple[ProofStep, ...], root: str
) -> tuple[dict[str, ProofStep], set[str]]:
    by_id = {step.id: step for step in steps}
    if len(by_id) != len(steps) or root not in by_id:
        raise ValueError("proof steps must have unique IDs and a valid root")
    reachable: set[str] = set()
    todo = [root]
    while todo:
        step_id = todo.pop()
        if step_id in reachable:
            continue
        try:
            step = by_id[step_id]
        except KeyError as exc:
            raise ValueError(f"proof step references unknown input: {step_id}") from exc
        reachable.add(step_id)
        todo.extend(step.inputs)
    return by_id, reachable


def measure_proof_graph(
    steps: tuple[ProofStep, ...], root: str
) -> ProofGraphMeasurement:
    """Measure one replay proof using only nodes reachable from its conclusion."""
    by_id, reachable = _reachable_steps(steps, root)
    semantic_depth: dict[str, int] = {}
    full_depth: dict[str, int] = {}
    signatures: dict[str, set[tuple[str, ...]]] = {}

    def visit(step_id: str) -> None:
        if step_id in semantic_depth:
            return
        step = by_id[step_id]
        for input_id in step.inputs:
            visit(input_id)
        input_semantic = max(
            (semantic_depth[input_id] for input_id in step.inputs), default=0
        )
        input_full = max((full_depth[input_id] for input_id in step.inputs), default=0)
        is_semantic = step.rule in SEMANTIC_RULES
        semantic_depth[step_id] = input_semantic + int(is_semantic)
        full_depth[step_id] = input_full + int(step.rule not in _NON_INFERENTIAL_RULES)

        longest = [
            input_id
            for input_id in step.inputs
            if semantic_depth[input_id] == input_semantic
        ]
        prefixes = {
            signature
            for input_id in longest
            for signature in signatures[input_id]
        } or {()}
        label = step.rule.value if is_semantic else None
        signatures[step_id] = {
            (*prefix, label) if label is not None else prefix for prefix in prefixes
        }

    visit(root)
    semantic_steps = [
        by_id[step_id]
        for step_id in reachable
        if by_id[step_id].rule in SEMANTIC_RULES
    ]
    operator_counts = Counter(step.rule.value for step in semantic_steps)
    critical = tuple(sorted(signatures[root]))
    transitions = tuple(
        sorted(
            {
                (first, second)
                for signature in critical
                for first, second in zip(signature, signature[1:])
            }
        )
    )
    support = tuple(
        sorted(
            {
                by_id[step_id].premise_index
                for step_id in reachable
                if by_id[step_id].rule is ProofRule.PREMISE
                and by_id[step_id].premise_index is not None
            }
        )
    )
    return ProofGraphMeasurement(
        semantic_length=semantic_depth[root],
        full_depth=full_depth[root],
        semantic_size=len(semantic_steps),
        branch_steps=sum(step.rule is ProofRule.CASE_SPLIT for step in semantic_steps),
        merge_steps=sum(len(step.inputs) > 1 for step in semantic_steps),
        support_premise_indices=support,
        operator_counts=dict(sorted(operator_counts.items())),
        critical_signatures=critical,
        ordered_transitions=transitions,
    )


def _proof_objects(value: Any) -> Iterable[tuple[tuple[ProofStep, ...], str]]:
    """Yield replay proof DAGs nested anywhere inside an answer certificate."""
    seen: set[int] = set()

    def walk(item: Any) -> Iterable[tuple[tuple[ProofStep, ...], str]]:
        if item is None or isinstance(item, (str, int, float, bool, bytes)):
            return
        identifier = id(item)
        if identifier in seen:
            return
        seen.add(identifier)
        if isinstance(item, DirectionProofCertificate):
            yield item.steps, item.conclusion_step
            return
        if isinstance(item, DirectionRefutationCertificate):
            yield item.steps, item.contradiction_step
            return
        if isinstance(item, FormulaRefutationCertificate):
            yield item.steps, item.contradiction_step
            return
        if isinstance(item, dict):
            for child in item.values():
                yield from walk(child)
            return
        if isinstance(item, (tuple, list, set, frozenset)):
            for child in item:
                yield from walk(child)
            return
        if is_dataclass(item):
            for field in fields(item):
                if field.name == "problem":
                    continue
                yield from walk(getattr(item, field.name))

    yield from walk(value)


def measure_certificate_complexity(
    problem: SpatialProblem,
    certificate: Any,
    *,
    removable_premises: int | None = None,
) -> ProofComplexityProfile:
    """Aggregate input and proof measurements over a complete answer certificate."""
    top_level = problem.premise.operands if isinstance(problem.premise, And) else (
        problem.premise,
    )
    atoms: list[RelationConstraint] = []
    connective_counts: Counter[str] = Counter()
    nesting = 0
    for premise in top_level:
        child_atoms, child_counts, child_depth = _formula_measurements(premise)
        atoms.extend(child_atoms)
        connective_counts.update(child_counts)
        nesting = max(nesting, child_depth)

    measurements = [
        measure_proof_graph(steps, root) for steps, root in _proof_objects(certificate)
    ]
    operators: Counter[str] = Counter()
    support: set[int] = set()
    signatures: set[tuple[str, ...]] = set()
    transitions: set[tuple[str, str]] = set()
    for measurement in measurements:
        operators.update(measurement.operator_counts)
        support.update(measurement.support_premise_indices)
        signatures.update(measurement.critical_signatures)
        transitions.update(measurement.ordered_transitions)

    removable = (
        max(0, len(top_level) - len(support))
        if removable_premises is None
        else removable_premises
    )
    return ProofComplexityProfile(
        top_level_premises=len(top_level),
        atomic_occurrences=len(atoms),
        unique_atoms=len(set(atoms)),
        connective_counts=dict(sorted(connective_counts.items())),
        formula_nesting=nesting,
        semantic_length=max(
            (measurement.semantic_length for measurement in measurements), default=0
        ),
        full_certificate_depth=max(
            (measurement.full_depth for measurement in measurements), default=0
        ),
        semantic_proof_size=sum(
            measurement.semantic_size for measurement in measurements
        ),
        branch_steps=sum(measurement.branch_steps for measurement in measurements),
        merge_steps=sum(measurement.merge_steps for measurement in measurements),
        proof_obligations=len(measurements),
        support_premise_indices=tuple(sorted(support)),
        supporting_premises=len(support),
        removable_premises=removable,
        operator_counts=dict(sorted(operators.items())),
        critical_signatures=tuple(sorted(signatures)),
        ordered_transitions=tuple(sorted(transitions)),
    )
