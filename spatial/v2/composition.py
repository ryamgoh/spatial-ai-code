"""Balanced logical-spatial compositions with checked ordered dependencies."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from spatial.v2.proofs import (
    ProofConstructionError,
    ProofRule,
    build_direction_proof,
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
    SpatialProblem,
    SpatialSolverV2,
    WhichQuery,
    direction_constraint,
)

# These are construction families, not inferred labels for arbitrary formulas.
COMPOSITIONS = {
    "mp-trc": ("direction", (ProofRule.MODUS_PONENS, ProofRule.AXIS_TRANSITIVITY)),
    "iff-trc": ("direction", (ProofRule.IFF_ELIMINATION, ProofRule.AXIS_TRANSITIVITY)),
    "dje-trc": (
        "direction",
        (ProofRule.DISJUNCTIVE_SYLLOGISM, ProofRule.AXIS_TRANSITIVITY),
    ),
    "trc-mp": ("direction", (ProofRule.AXIS_TRANSITIVITY, ProofRule.MODUS_PONENS)),
    "mp-trc-mp": (
        "direction",
        (ProofRule.MODUS_PONENS, ProofRule.AXIS_TRANSITIVITY, ProofRule.MODUS_PONENS),
    ),
    "mp-dje-trc": (
        "direction",
        (
            ProofRule.MODUS_PONENS,
            ProofRule.DISJUNCTIVE_SYLLOGISM,
            ProofRule.AXIS_TRANSITIVITY,
        ),
    ),
    "trc-mp-trc": (
        "direction",
        (
            ProofRule.AXIS_TRANSITIVITY,
            ProofRule.MODUS_PONENS,
            ProofRule.AXIS_TRANSITIVITY,
        ),
    ),
}
TRAIN_COMPOSITIONS = ("mp-trc", "iff-trc", "trc-mp", "dje-trc")
HOLDOUT_COMPOSITIONS = ("mp-trc-mp", "mp-dje-trc", "trc-mp-trc")


def composition_structures(name: str) -> tuple[str, ...]:
    return tuple(part.upper() for part in name.split("-"))


def validate_composition_policy(policy: Any) -> None:
    if policy.composition not in COMPOSITIONS:
        raise ValueError(f"unknown composition: {policy.composition}")
    if (
        policy.semantic_shape.value != "unique"
        or policy.boolean_shape.value != "atomic"
    ):
        raise ValueError(
            "composition policies require unique semantics and atomic base shape"
        )
    if policy.generation_mode.value not in {"auto", "proof-template-first"}:
        raise ValueError(
            "composition policies require proof-template-first or auto mode"
        )
    if (
        policy.query_kind.value == "which"
        and policy.query_direction is not None
        and policy.query_direction
        not in {
            Direction.NORTHEAST,
            Direction.NORTHWEST,
            Direction.SOUTHEAST,
            Direction.SOUTHWEST,
        }
    ):
        raise ValueError("Selection composition requires a diagonal query direction")
    if policy.composition == "mp-trc-mp" and policy.num_entities < 7:
        raise ValueError(
            "mp-trc-mp requires seven objects for independent trigger support"
        )
    if policy.num_entities < 5 or policy.num_premises < policy.num_entities - 1:
        raise ValueError(
            "compositions require at least five objects and a connected premise budget"
        )
    if (
        policy.target_direction is not None
        or policy.omit_direct_query_relation
        or policy.min_axis_depth != 1
        or policy.max_axis_depth is not None
        or policy.min_membership_depth != 1
        or policy.max_membership_depth is not None
        or policy.require_independent_axes
        or policy.distractor_premises
        or policy.require_joint_count_dependency
    ):
        raise ValueError(
            "composition structure cannot be crossed with independent proof controls"
        )


def construct_composition(
    policy: Any, objects: tuple[str, ...], rng: Any
) -> SpatialProblem:
    """Instantiate one declared two-stage or interleaved support skeleton."""
    nodes = list(objects)
    rng.shuffle(nodes)
    target, intermediate, reference, first, second, *extras = nodes
    if policy.composition in {"mp-trc-mp", "trc-mp-trc"} and not extras:
        raise ValueError("three-stage compositions require at least six objects")
    direction = policy.query_direction or rng.choice(tuple(Direction))
    if policy.query_kind.value == "which":
        direction = policy.query_direction or rng.choice(
            (
                Direction.NORTHEAST,
                Direction.NORTHWEST,
                Direction.SOUTHEAST,
                Direction.SOUTHWEST,
            )
        )
    atom = direction_constraint(target, intermediate, direction)
    trigger = direction_constraint(first, second, rng.choice(tuple(Direction)))
    alternative = direction_constraint(
        target,
        intermediate,
        rng.choice(tuple(d for d in Direction if d is not direction)),
    )
    name = policy.composition
    if name == "mp-trc":
        support = [trigger, Implies(trigger, atom)]
    elif name == "iff-trc":
        support = [trigger, Iff(trigger, atom)]
    elif name == "dje-trc":
        support = [Or((atom, alternative)), Not(alternative)]
    elif name in {"trc-mp", "mp-trc-mp", "trc-mp-trc"}:
        antecedent = direction_constraint(first, second, Direction.EAST)
        pivot = intermediate if name in {"trc-mp", "mp-trc-mp"} else extras[0]
        first_edge = direction_constraint(first, pivot, Direction.EAST)
        second_edge = direction_constraint(pivot, second, Direction.EAST)
        if name == "mp-trc-mp":
            trigger = direction_constraint(extras[0], extras[1], Direction.NORTH)
        support = (
            [trigger, Implies(trigger, first_edge), second_edge]
            if name == "mp-trc-mp"
            else [first_edge, second_edge]
        )
        result = (
            direction_constraint(target, reference, direction)
            if name != "trc-mp-trc"
            else atom
        )
        support.append(Implies(antecedent, result))
    elif name == "mp-dje-trc":
        support = [trigger, Implies(trigger, Or((atom, alternative))), Not(alternative)]
    else:
        raise ValueError(f"unknown composition: {name}")
    second_direction = (
        (
            Direction.NORTH
            if direction in {Direction.NORTHEAST, Direction.NORTHWEST}
            else Direction.SOUTH
        )
        if policy.query_kind.value == "which"
        else direction
    )
    premises = list(support)
    if name not in {"trc-mp", "mp-trc-mp"}:
        premises.append(direction_constraint(intermediate, reference, second_direction))
    elif policy.query_kind.value != "direction":
        # Keep all named candidate memberships determined without changing the query.
        premises.append(Not(direction_constraint(intermediate, reference, direction)))
    if policy.query_kind.value == "direction":
        # Connect remaining objects without adding a target/reference shortcut.
        premises.extend(
            Or(
                (
                    direction_constraint(extra, reference, Direction.NORTHWEST),
                    Not(direction_constraint(extra, reference, Direction.NORTHWEST)),
                )
            )
            for extra in (first, second, *extras)
        )
        query = DirectionQuery(target, reference)
    else:
        # The intermediate is support-only; the query lists its candidate domain.
        candidates = tuple(obj for obj in objects if obj != reference)
        premises.extend(
            Not(direction_constraint(obj, reference, direction))
            for obj in (first, second, *extras)
        )
        query_type = WhichQuery if policy.query_kind.value == "which" else CountQuery
        query = query_type(frozenset({direction}), reference, candidates)
    if len(premises) > policy.num_premises:
        raise ProofConstructionError("composition premise budget is too small")
    # Budget fillers are explicit tautologies and never carry query support.
    for obj in objects:
        if len(premises) == policy.num_premises:
            break
        if obj == reference:
            continue
        filler = direction_constraint(obj, reference, Direction.EAST)
        premises.append(Or((filler, Not(filler))))
    if len(premises) != policy.num_premises:
        raise ProofConstructionError(
            "composition premise budget exceeds filler capacity"
        )
    rng.shuffle(premises)
    return SpatialProblem(objects, And(tuple(premises)), query)


def assess_composition(
    problem: SpatialProblem, name: str, solver: SpatialSolverV2
) -> dict[str, Any]:
    """Check ordered rule use and query change after removing its premise support.

    This is a bounded intervention, not a claim of globally shortest or unique
    proof. Dependency order is checked in a replay-valid positive proof.
    """
    if name not in COMPOSITIONS:
        raise ProofConstructionError(f"unknown composition: {name}")
    _, required_rules = COMPOSITIONS[name]
    actual_kind = (
        "direction"
        if isinstance(problem.query, DirectionQuery)
        else "which"
        if isinstance(problem.query, WhichQuery)
        else "count"
    )
    kind = actual_kind
    analysis = solver.analyze(problem)
    if analysis.error or not analysis.consistent:
        raise ProofConstructionError("composition premises must be consistent")
    query = problem.query
    if isinstance(query, DirectionQuery):
        if len(analysis.possible_directions) != 1:
            raise ProofConstructionError("composition requires one entailed direction")
        projections = [(query.target, analysis.possible_directions[0])]
    else:
        direction = next(iter(query.directions))
        projections = [
            (candidate, direction)
            for candidate in query.candidates
            if solver.assess(
                problem, direction_constraint(candidate, query.reference, direction)
            ).entailed
        ]
    for target, direction in projections:
        projected = replace(problem, query=DirectionQuery(target, query.reference))
        proof = build_direction_proof(projected, direction)
        by_id = {step.id: step for step in proof.steps}

        def ancestors(step_id: str, steps_by_id=by_id) -> set[str]:
            found: set[str] = set()
            todo = list(steps_by_id[step_id].inputs)
            while todo:
                current = todo.pop()
                if current in found:
                    continue
                found.add(current)
                todo.extend(steps_by_id[current].inputs)
            return found

        # An ordered chain requires distinct steps, each preceding step a true
        # ancestor of the next. Merely listing rules in one proof is insufficient.
        chains = [()]
        for rule in required_rules:
            chains = [
                (*chain, step.id)
                for chain in chains
                for step in proof.steps
                if step.rule is rule and (not chain or chain[-1] in ancestors(step.id))
            ]
        if not chains:
            continue
        chain = min(chains)
        linked = [by_id[identifier] for identifier in chain]
        support_indices = sorted(
            {
                by_id[ancestor].premise_index
                for step in linked
                for ancestor in ancestors(step.id)
                if by_id[ancestor].rule is ProofRule.PREMISE
            }
        )
        premises = (
            problem.premise.operands
            if isinstance(problem.premise, And)
            else (problem.premise,)
        )
        # Test individual support interventions; retaining a shortcut fails admission.
        interventions = []
        for index in support_indices:
            reduced = replace(
                problem,
                premise=And(tuple(p for i, p in enumerate(premises) if i != index)),
            )
            after = solver.analyze(reduced)
            if after.error or not after.consistent:
                raise ProofConstructionError(
                    "composition intervention failed semantic checking"
                )
            changed = (
                after.possible_directions != analysis.possible_directions
                if kind == "direction"
                else (after.possible_entities, after.entailed_entities)
                != (analysis.possible_entities, analysis.entailed_entities)
                if kind == "which"
                else after.possible_counts != analysis.possible_counts
            )
            if not changed:
                raise ProofConstructionError(
                    "composition support can be bypassed without changing query semantics"
                )
            interventions.append(index)
        if interventions:
            return {
                "schema": "logical-spatial-composition-v1",
                "skeleton": name,
                "required_rules": [rule.value for rule in required_rules],
                "linked_steps": [step.id for step in linked],
                "intervention_premise_indices": interventions,
                "claim": "replay-valid ordered rule chain with individual support-removal dependence",
            }
    raise ProofConstructionError(
        "no positive query proof links the declared logical and spatial rules"
    )
