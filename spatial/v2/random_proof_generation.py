"""Construct solver-labelled spatial problems from randomized proof backbones."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations

from spatial.v2.certificate_generation import AnswerCertificate, build_answer_certificate
from spatial.v2.proof_complexity import ProofComplexityProfile, measure_certificate_complexity
from spatial.v2.random_proof import (
    PROOF_RULE_LABELS,
    RandomProofPolicy,
    SemanticRule,
    sample_policy_path,
)
from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    Iff,
    Implies,
    Not,
    Or,
    QueryAnalysis,
    RelationConstraint,
    SpatialFormula,
    SpatialProblem,
    SpatialSolverV2,
    WhichQuery,
    direction_constraint,
    direction_signs,
)
from spatial.v2.text import spatial_formula_objects


class RandomProofConstructionError(ValueError):
    """The sampled backbone cannot be instantiated in the supported fragment."""


@dataclass(frozen=True)
class RandomProofCandidate:
    problem: SpatialProblem
    analysis: QueryAnalysis
    certificate: AnswerCertificate
    requested_path: tuple[SemanticRule, ...]
    complexity: ProofComplexityProfile


@dataclass(frozen=True)
class LiftedProofProblem:
    problem: SpatialProblem
    requested_path: tuple[SemanticRule, ...]


def possibility_backbone_length(
    query_kind: str, skeleton: str, gold_size: int
) -> int | None:
    """Return a constructive L offset; final certificates still verify actual L.

    Randomized v2 uses exclusion-first Direction/Selection skeletons and a DJE
    Count skeleton. Their complete certificates have stable query-specific
    backbones, so generation need not rebuild a base certificate on every
    rejected lift merely to estimate the prefix size.
    """
    if skeleton == "exclude":
        return 0 if query_kind in {"direction", "which"} else None
    if skeleton != "dje":
        return None
    if query_kind == "direction":
        return 1
    if query_kind == "which":
        return 0
    if query_kind == "count":
        return gold_size
    raise ValueError(f"unknown query kind: {query_kind}")


class _AtomFactory:
    def __init__(
        self,
        objects: tuple[str, ...],
        rng: random.Random,
        *,
        preferred_objects: tuple[str, ...] = (),
    ) -> None:
        preferred_pairs = list(combinations(preferred_objects, 2))
        preferred_keys = {frozenset(pair) for pair in preferred_pairs}
        pairs = [
            *preferred_pairs,
            *(
                pair
                for pair in combinations(objects, 2)
                if frozenset(pair) not in preferred_keys
            ),
        ]
        rng.shuffle(pairs)
        directions = list(Direction)
        rng.shuffle(directions)
        self._atoms = [
            direction_constraint(first, second, directions[index % len(directions)])
            for index, (first, second) in enumerate(pairs)
        ]
        self._cursor = 0

    def next(self) -> RelationConstraint:
        if self._cursor >= len(self._atoms):
            raise RandomProofConstructionError(
                "not enough distinct atoms for proof graph"
            )
        atom = self._atoms[self._cursor]
        self._cursor += 1
        return atom


def _single_direction(atom: RelationConstraint) -> Direction:
    if len(atom.allowed) != 1:
        raise RandomProofConstructionError("TRC requires an exact spatial atom")
    return next(iter(atom.allowed))


def _backward_step(
    rule: SemanticRule,
    output: SpatialFormula,
    atoms: _AtomFactory,
    objects: tuple[str, ...],
    rng: random.Random,
    side_premises: list[SpatialFormula],
    preferred_objects: tuple[str, ...] = (),
) -> SpatialFormula:
    if rule is SemanticRule.MP:
        source = atoms.next()
        side_premises.append(Implies(source, output))
        return source
    if rule is SemanticRule.IFF:
        source = atoms.next()
        side_premises.append(Iff(source, output))
        return source
    if rule is SemanticRule.DJE:
        eliminated = atoms.next()
        side_premises.append(Or((eliminated, output)))
        return Not(eliminated)
    if rule is SemanticRule.DN:
        return Not(Not(output))
    if rule is SemanticRule.TRC:
        if not isinstance(output, RelationConstraint):
            raise RandomProofConstructionError(
                "TRC cannot produce a compound Boolean formula"
            )
        direction = _single_direction(output)
        candidates = [
            obj
            for obj in preferred_objects
            if obj not in {output.subject, output.reference}
        ]
        fallback = [
            obj
            for obj in objects
            if obj not in {output.subject, output.reference}
            and obj not in candidates
        ]
        rng.shuffle(candidates)
        rng.shuffle(fallback)
        candidates.extend(fallback)
        if not candidates:
            raise RandomProofConstructionError("TRC needs an intermediate object")
        intermediate = candidates[0]
        source = direction_constraint(output.subject, intermediate, direction)
        side_premises.append(
            direction_constraint(intermediate, output.reference, direction)
        )
        return source
    raise AssertionError(f"unsupported semantic rule: {rule}")


def _is_padding_tautology(formula: SpatialFormula) -> bool:
    if not isinstance(formula, Or) or len(formula.operands) != 2:
        return False
    first, second = formula.operands
    return (isinstance(second, Not) and second.operand == first) or (
        isinstance(first, Not) and first.operand == second
    )


def _pruned_problem(
    objects: tuple[str, ...],
    premise: SpatialFormula,
    query: DirectionQuery | WhichQuery | CountQuery,
) -> SpatialProblem:
    used = set(spatial_formula_objects(premise))
    if isinstance(query, DirectionQuery):
        used.update((query.target, query.reference))
    else:
        used.add(query.reference)
        used.update(query.candidates)
    return SpatialProblem(tuple(obj for obj in objects if obj in used), premise, query)


def _proof_and_distractor_objects(
    objects: tuple[str, ...],
    mentioned: set[str],
    distractor_count: int,
    rng: random.Random,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    spare = [obj for obj in objects if obj not in mentioned]
    rng.shuffle(spare)
    if len(spare) < distractor_count:
        raise RandomProofConstructionError(
            "not enough unused objects for atomic leaf distractors"
        )
    leaves = tuple(spare[:distractor_count])
    proof_pool = spare[distractor_count:]
    # Five objects provide ten distinct pairs, enough for an L8 linear path.
    # Reusing this bounded core avoids bloating the solver domain merely to
    # obtain fresh propositional atoms.
    proof_core = tuple(proof_pool[:5])
    atom_objects = tuple(obj for obj in objects if obj not in leaves)
    return atom_objects, proof_core


def _attached_leaf_distractors(
    leaves: tuple[str, ...], anchors: set[str], rng: random.Random
) -> list[RelationConstraint]:
    attached = list(sorted(anchors))
    distractors = []
    for leaf in leaves:
        if not attached:
            raise RandomProofConstructionError("leaf distractor needs one anchor")
        distractors.append(
            direction_constraint(
                leaf,
                rng.choice(attached),
                rng.choice(tuple(Direction)),
            )
        )
        attached.append(leaf)
    return distractors


def lift_problem_with_random_path(
    problem: SpatialProblem,
    policy: RandomProofPolicy,
    *,
    seed: int,
) -> LiftedProofProblem:
    """Replace one required atomic premise with a sampled derivation backbone."""
    rng = random.Random(seed)
    path = sample_policy_path(rng, policy)
    top_level = list(
        problem.premise.operands
        if isinstance(problem.premise, And)
        else (problem.premise,)
    )
    top_level = [item for item in top_level if not _is_padding_tautology(item)]
    mentioned = {
        name
        for premise in top_level
        for name in spatial_formula_objects(premise)
    }
    if isinstance(problem.query, DirectionQuery):
        mentioned.update((problem.query.target, problem.query.reference))
    else:
        mentioned.add(problem.query.reference)
        mentioned.update(problem.query.candidates)
    exact_indices = [
        index
        for index, premise in enumerate(top_level)
        if isinstance(premise, RelationConstraint) and len(premise.allowed) == 1
    ]
    if path and path[-1] is SemanticRule.TRC:
        target_indices = exact_indices
    elif isinstance(problem.query, WhichQuery):
        target_indices = [
            index
            for index, premise in enumerate(top_level)
            if isinstance(premise, Not)
            and isinstance(premise.operand, RelationConstraint)
        ] or list(range(len(top_level)))
    else:
        target_indices = list(range(len(top_level)))
    if not target_indices:
        raise RandomProofConstructionError(
            "possibility skeleton has no compatible premise to lift"
        )
    target_index = rng.choice(target_indices)
    target = top_level.pop(target_index)
    atom_objects, proof_core = _proof_and_distractor_objects(
        problem.objects, mentioned, policy.distractor_count, rng
    )
    leaves = tuple(obj for obj in problem.objects if obj not in atom_objects)
    atoms = _AtomFactory(atom_objects, rng, preferred_objects=proof_core)
    side_premises: list[SpatialFormula] = []
    current: SpatialFormula = target
    for rule in reversed(path):
        current = _backward_step(
            rule,
            current,
            atoms,
            atom_objects,
            rng,
            side_premises,
            proof_core,
        )
    top_level.extend((current, *side_premises))
    top_level.extend(_attached_leaf_distractors(leaves, mentioned, rng))
    rng.shuffle(top_level)
    premise = And(tuple(top_level))
    return LiftedProofProblem(
        _pruned_problem(problem.objects, premise, problem.query), path
    )


def _query_problem(
    query_kind: str,
    objects: tuple[str, ...],
    rng: random.Random,
) -> tuple[RelationConstraint, DirectionQuery | WhichQuery | CountQuery, list[SpatialFormula]]:
    if len(objects) < 8:
        raise RandomProofConstructionError("random proof graphs require eight objects")
    target, reference, *rest = objects
    direction = rng.choice(tuple(Direction))
    final = direction_constraint(target, reference, direction)
    if query_kind == "direction":
        return final, DirectionQuery(target, reference), []
    candidates = (target, *rest[:4])
    signs = direction_signs(direction)
    opposite = next(
        candidate
        for candidate in Direction
        if direction_signs(candidate) == (-signs[0], -signs[1])
    )
    exclusions = [
        direction_constraint(candidate, reference, opposite)
        for candidate in candidates
        if candidate != target
    ]
    if query_kind == "which":
        return final, WhichQuery(frozenset({direction}), reference, candidates), exclusions
    if query_kind == "count":
        return final, CountQuery(frozenset({direction}), reference, candidates), exclusions
    raise ValueError(f"unknown query kind: {query_kind}")


def construct_random_proof_candidate(
    *,
    query_kind: str,
    objects: tuple[str, ...],
    policy: RandomProofPolicy,
    seed: int,
    solver: SpatialSolverV2 | None = None,
) -> RandomProofCandidate:
    """Construct and certify one unique-answer candidate with an exact backbone."""
    rng = random.Random(seed)
    solver = solver or SpatialSolverV2()
    final, query, exclusions = _query_problem(query_kind, objects, rng)
    path = sample_policy_path(rng, policy)
    mentioned = {
        name
        for formula in (final, *exclusions)
        for name in spatial_formula_objects(formula)
    }
    atom_objects, proof_core = _proof_and_distractor_objects(
        objects, mentioned, policy.distractor_count, rng
    )
    leaves = tuple(obj for obj in objects if obj not in atom_objects)
    atom_factory = _AtomFactory(atom_objects, rng, preferred_objects=proof_core)
    side_premises: list[SpatialFormula] = []
    current: SpatialFormula = final
    for rule in reversed(path):
        current = _backward_step(
            rule,
            current,
            atom_factory,
            atom_objects,
            rng,
            side_premises,
            proof_core,
        )
    premises: list[SpatialFormula] = [current, *side_premises, *exclusions]
    premises.extend(_attached_leaf_distractors(leaves, mentioned, rng))
    rng.shuffle(premises)
    premise = And(tuple(premises))
    problem = _pruned_problem(objects, premise, query)
    analysis = solver.analyze(problem)
    if analysis.error or not analysis.consistent:
        raise RandomProofConstructionError(
            analysis.error or "constructed proof graph is inconsistent"
        )
    try:
        certificate = build_answer_certificate(problem, analysis, solver)
    except ValueError as exc:
        raise RandomProofConstructionError(str(exc)) from exc
    complexity = measure_certificate_complexity(problem, certificate)
    requested = tuple(PROOF_RULE_LABELS[rule] for rule in path)
    if policy.semantic_length and requested not in complexity.critical_signatures:
        raise RandomProofConstructionError(
            "replayed certificate does not contain the requested critical path"
        )
    if complexity.semantic_length != policy.semantic_length:
        raise RandomProofConstructionError(
            f"requested L{policy.semantic_length} but realised "
            f"L{complexity.semantic_length}"
        )
    if complexity.removable_premises != policy.distractor_count:
        raise RandomProofConstructionError(
            f"requested X{policy.distractor_count} but realised "
            f"X{complexity.removable_premises}"
        )
    return RandomProofCandidate(
        problem,
        analysis,
        certificate,
        path,
        complexity,
    )
