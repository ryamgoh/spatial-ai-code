"""Proof-template constructions whose answers are possibility sets.

`SINGLE` collapses every non-unique answer to `Cannot be determined`, so its
templates only need one invariant result. These skeletons instead plant two or
more possible answers behind the same depth, Boolean, joint-count, and
distractor structures used by the unique cells. Each one returns a candidate
problem; the generator still solves it, matches the declared possibility count,
and builds a replayable certificate before admission.
"""

from __future__ import annotations

from itertools import combinations, pairwise
from typing import Any

from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    Iff,
    Implies,
    Not,
    Or,
    SpatialFormula,
    SpatialProblem,
    WhichQuery,
    direction_constraint,
    direction_signs,
)

_DIRECTION_BY_SIGNS = {direction_signs(value): value for value in Direction}
_DIAGONALS = (
    Direction.NORTHEAST,
    Direction.SOUTHEAST,
    Direction.SOUTHWEST,
    Direction.NORTHWEST,
)

# skeleton -> (query kinds, minimum objects)
POSSIBILITY_SKELETONS = {
    "depth": (frozenset({"direction", "which", "count"}), 4),
    "mp": (frozenset({"direction", "which", "count"}), 5),
    "iff": (frozenset({"direction", "which", "count"}), 5),
    "dje": (frozenset({"direction", "which", "count"}), 5),
    "csr": (frozenset({"direction", "which", "count"}), 5),
    "jcr": (frozenset({"count"}), 5),
    "noise": (frozenset({"direction", "which"}), 5),
    "case-chain": (frozenset({"which"}), 13),
    "case-count": (frozenset({"count"}), 8),
    "factor-count": (frozenset({"count"}), 10),
    "exclude": (frozenset({"direction", "which"}), 6),
}


# Reasoning-structure codes each skeleton exercises, for registry validation.
POSSIBILITY_STRUCTURES = {
    "depth": ("TRC",),
    "mp": ("MP",),
    "iff": ("IFF",),
    "dje": ("DJE",),
    "csr": ("CSR",),
    "jcr": ("CSR",),
    "noise": ("TRC",),
    "case-chain": ("CSR", "TRC"),
    "case-count": ("CSR",),
    "factor-count": ("IFF",),
    "exclude": ("NEG",),
}


class PossibilityConstructionError(ValueError):
    pass


def validate_possibility_policy(policy: Any) -> None:
    if policy.possibility_skeleton not in POSSIBILITY_SKELETONS:
        raise ValueError(f"unknown possibility skeleton: {policy.possibility_skeleton}")
    kinds, minimum = POSSIBILITY_SKELETONS[policy.possibility_skeleton]
    if policy.query_kind.value not in kinds:
        raise ValueError(
            f"{policy.possibility_skeleton} possibility skeleton does not support "
            f"{policy.query_kind.value} queries"
        )
    if policy.semantic_shape.value != "ambiguous":
        raise ValueError("possibility skeletons require ambiguous semantics")
    if policy.boolean_shape.value != "atomic" or policy.composition is not None:
        raise ValueError("possibility skeletons declare their own Boolean structure")
    if policy.num_entities < minimum:
        raise ValueError(
            f"{policy.possibility_skeleton} possibility skeleton requires at least "
            f"{minimum} objects"
        )
    if policy.possibility_depth < 1:
        raise ValueError("possibility_depth must be positive")
    gold_size = policy.possibility_gold_size
    # A fixed five-option menu must show the complete possible set plus at least
    # one distractor, so the planted gold set can hold at most four answers.
    if policy.possibility_skeleton in {
        "mp",
        "iff",
        "dje",
        "csr",
        "jcr",
        "case-chain",
        "case-count",
        "factor-count",
        "exclude",
    }:
        if not 2 <= gold_size <= 4:
            raise ValueError("possibility_gold_size must be between 2 and 4")
        if (
            policy.possibility_skeleton in {"mp", "iff", "dje", "csr", "jcr"}
            and policy.query_kind.value != "direction"
        ):
            needed = 3 + gold_size + (1 if policy.possibility_skeleton == "dje" else 0)
            if policy.num_entities < needed:
                raise ValueError(
                    f"{policy.possibility_skeleton} with gold size {gold_size} "
                    f"requires at least {needed} objects"
                )
        if (
            policy.possibility_skeleton == "exclude"
            and policy.query_kind.value == "count"
        ):
            raise ValueError("exclude does not define correlated Count possibilities")
        if policy.possibility_skeleton == "case-chain":
            needed = 8 + 5 * (policy.possibility_depth - 1)
            if policy.num_entities < needed:
                raise ValueError(
                    f"case-chain with depth {policy.possibility_depth} requires "
                    f"at least {needed} objects"
                )
            minimum_premises = (
                1
                + 5 * policy.possibility_depth
                + policy.possibility_gold_size
            )
            if policy.num_premises < minimum_premises:
                raise ValueError(
                    f"case-chain with depth {policy.possibility_depth} and gold "
                    f"size {gold_size} requires at least {minimum_premises} premises"
                )
        if policy.possibility_skeleton == "case-count" and policy.num_entities < 8:
            raise ValueError("case-count requires at least 8 objects")
        if policy.possibility_skeleton == "factor-count":
            if gold_size not in {2, 3, 4}:
                raise ValueError("factor-count supports gold size 2, 3, or 4")
            if policy.num_entities < 10 or policy.num_premises < 9:
                raise ValueError(
                    "factor-count requires at least 10 objects and 9 premises"
                )
    elif gold_size != 2:
        raise ValueError(
            "possibility_gold_size applies only to Boolean and jcr skeletons"
        )
    if (
        policy.possibility_skeleton in {"depth", "noise", "case-chain"}
        and policy.possibility_depth < 2
    ):
        raise ValueError(
            "depth, noise, and case-chain skeletons require a chain of at least 2"
        )
    if policy.possibility_skeleton == "noise" and policy.distractor_premises < 1:
        raise ValueError("noise possibility skeleton requires distractor premises")
    if policy.possibility_skeleton != "noise" and policy.distractor_premises:
        raise ValueError("only the noise possibility skeleton adds distractors")
    if (
        policy.target_direction is not None
        or policy.omit_direct_query_relation
        or policy.min_axis_depth != 1
        or policy.max_axis_depth is not None
        or policy.min_membership_depth != 1
        or policy.max_membership_depth is not None
        or policy.require_independent_axes
        or policy.require_joint_count_dependency
    ):
        raise ValueError(
            "possibility skeletons cannot be crossed with unique proof controls"
        )


def construct_possibility(
    policy: Any, objects: tuple[str, ...], rng: Any
) -> SpatialProblem:
    """Instantiate one declared possibility skeleton."""
    nodes = list(objects)
    rng.shuffle(nodes)
    kind = policy.query_kind.value
    name = policy.possibility_skeleton
    if kind == "direction":
        premises, query, used = _direction_skeleton(name, policy, nodes, rng)
    else:
        premises, query, used = _membership_skeleton(name, policy, nodes, rng)
    if len(premises) > policy.num_premises:
        raise PossibilityConstructionError("possibility premise budget is too small")
    premises.extend(_tautologies(objects, policy.num_premises - len(premises), rng))
    if len(premises) != policy.num_premises:
        raise PossibilityConstructionError(
            "possibility premise budget exceeds filler capacity"
        )
    rng.shuffle(premises)
    return SpatialProblem(objects, And(tuple(premises)), query)


def _direction_skeleton(
    name: str, policy: Any, nodes: list[str], rng: Any
) -> tuple[list[SpatialFormula], DirectionQuery, set[str]]:
    target, reference, first, second, *extras = nodes
    query = DirectionQuery(target, reference)
    if name == "exclude":
        alternatives = tuple(rng.sample(list(Direction), policy.possibility_gold_size))
        premises = [
            Not(direction_constraint(target, reference, value))
            for value in Direction
            if value not in alternatives
        ]
        return premises, query, {target, reference}
    if name in {"depth", "noise"}:
        chain = [reference, *extras[: policy.possibility_depth - 1], target]
        if len(chain) != policy.possibility_depth + 1:
            raise PossibilityConstructionError("not enough objects for the chain")
        premises = _open_axis_chain(chain, rng)
        if name == "noise":
            premises.extend(
                _attached_distractors(
                    chain,
                    [first, second, *extras[policy.possibility_depth - 1 :]],
                    policy.distractor_premises,
                    rng,
                )
            )
        return premises, query, set(chain)
    trigger = direction_constraint(first, second, rng.choice(tuple(Direction)))
    alternatives = tuple(rng.sample(list(Direction), policy.possibility_gold_size))
    answers = Or(
        tuple(direction_constraint(target, reference, value) for value in alternatives)
    )
    if name == "mp":
        premises = [trigger, Implies(trigger, answers)]
    elif name == "iff":
        premises = [trigger, Iff(trigger, answers)]
    elif name == "dje":
        excluded = rng.choice(tuple(d for d in Direction if d not in alternatives))
        widened = Or(
            (
                *answers.operands,
                direction_constraint(target, reference, excluded),
            )
        )
        premises = [widened, Not(direction_constraint(target, reference, excluded))]
    elif name == "csr":
        cases = _exclusive_cases(first, second, len(alternatives), rng)
        premises = [
            Or(cases),
            *(
                Implies(case, direction_constraint(target, reference, value))
                for case, value in zip(cases, alternatives, strict=True)
            ),
        ]
    else:
        raise PossibilityConstructionError(f"{name} does not support Direction")
    return premises, query, {target, reference, first, second}


def _membership_skeleton(
    name: str, policy: Any, nodes: list[str], rng: Any
) -> tuple[list[SpatialFormula], WhichQuery | CountQuery, set[str]]:
    reference, *pool = nodes
    direction = policy.query_direction or rng.choice(_DIAGONALS)
    member = lambda candidate: direction_constraint(candidate, reference, direction)
    if name == "exclude":
        candidate_count = policy.possibility_candidate_count or 5
        if not policy.possibility_gold_size < candidate_count <= len(pool):
            raise PossibilityConstructionError(
                "exclude candidate count must exceed gold size and fit the pool"
            )
        candidates = pool[:candidate_count]
        planted = candidates[: policy.possibility_gold_size]
        others = candidates[policy.possibility_gold_size :]
        premises = [Not(member(candidate)) for candidate in others]
        rng.shuffle(candidates)
        return (
            premises,
            WhichQuery(frozenset({direction}), reference, tuple(candidates)),
            {reference, *planted, *others},
        )
    if name == "case-chain":
        first, second, *rest = pool
        candidates, intermediate_pool = rest[:5], rest[5:]
        cases = _exclusive_cases(first, second, policy.possibility_gold_size, rng)
        premises: list[SpatialFormula] = [Or(cases)]
        opposite = _opposite(direction)
        cursor = 0
        for index, candidate in enumerate(candidates):
            intermediate = intermediate_pool[
                cursor : cursor + policy.possibility_depth - 1
            ]
            cursor += policy.possibility_depth - 1
            chain = [reference, *intermediate, candidate]
            atoms = _exact_chain(
                chain, direction if index < policy.possibility_gold_size else opposite
            )
            if index < policy.possibility_gold_size:
                premises.append(Implies(cases[index], atoms[0]))
                premises.extend(atoms[1:])
                premises.append(
                    Implies(
                        Not(cases[index]),
                        direction_constraint(candidate, reference, opposite),
                    )
                )
            else:
                premises.extend(atoms)
        rng.shuffle(candidates)
        return (
            premises,
            WhichQuery(frozenset({direction}), reference, tuple(candidates)),
            {reference, first, second, *candidates, *intermediate_pool[:cursor]},
        )
    if name == "case-count":
        first, second, *rest = pool
        candidates = rest[:5]
        cases = _exclusive_cases(first, second, policy.possibility_gold_size, rng)
        counts = {
            2: (0, 5),
            3: (0, 2, 5),
            4: (0, 1, 3, 5),
        }[policy.possibility_gold_size]
        counts = tuple(rng.sample(counts, len(counts)))
        opposite = _opposite(direction)
        premises = [Or(cases)]
        for case, count in zip(cases, counts, strict=True):
            included = set(rng.sample(candidates, count))
            assignment = tuple(
                direction_constraint(
                    candidate,
                    reference,
                    direction if candidate in included else opposite,
                )
                for candidate in candidates
            )
            premises.append(Implies(case, And(assignment)))
        rng.shuffle(candidates)
        return (
            premises,
            CountQuery(frozenset({direction}), reference, tuple(candidates)),
            {reference, first, second, *candidates},
        )
    if name == "factor-count":
        first, second, third, fourth, *rest = pool
        candidates = rest[:5]
        triggers = (
            direction_constraint(first, second, rng.choice(tuple(Direction))),
            direction_constraint(third, fourth, rng.choice(tuple(Direction))),
        )
        group_sizes = {
            2: (5,),
            3: (2, 2),
            4: (2, 3),
        }[policy.possibility_gold_size]
        opposite = _opposite(direction)
        premises = []
        cursor = 0
        for trigger, group_size in zip(
            triggers[: len(group_sizes)], group_sizes, strict=True
        ):
            for _ in range(group_size):
                candidate = candidates[cursor]
                premises.append(Iff(trigger, member(candidate)))
                cursor += 1
        for candidate in candidates[cursor:]:
            premises.append(direction_constraint(candidate, reference, opposite))
        rng.shuffle(candidates)
        return (
            premises,
            CountQuery(frozenset({direction}), reference, tuple(candidates)),
            {reference, first, second, third, fourth, *candidates},
        )
    if name in {"depth", "noise"}:
        depth = policy.possibility_depth
        needed = 2 * (depth - 1) + 2
        if len(pool) < needed:
            raise PossibilityConstructionError("not enough objects for the chains")
        open_target, fixed_target = pool[0], pool[1]
        cursor = 2
        open_chain = [reference, *pool[cursor : cursor + depth - 1], open_target]
        cursor += depth - 1
        fixed_chain = [reference, *pool[cursor : cursor + depth - 1], fixed_target]
        cursor += depth - 1
        premises = [
            *_membership_chain(open_chain, direction, open_axis=True, rng=rng),
            *_membership_chain(fixed_chain, direction, open_axis=False, rng=rng),
        ]
        used = set(open_chain) | set(fixed_chain)
        candidates = [open_target, fixed_target]
        rest = pool[cursor:]
        if name == "noise":
            # Keep a real exclusion menu, then hang the remaining objects off
            # the support as removable distractors.
            excluded, spare = rest[:3], rest[3:]
            premises.extend(Not(member(candidate)) for candidate in excluded)
            candidates.extend(excluded)
            used.update(excluded)
            premises.extend(
                _attached_distractors(
                    sorted(used), spare, policy.distractor_premises, rng
                )
            )
        else:
            # Remaining objects are excluded candidates with checked refutations.
            premises.extend(Not(member(candidate)) for candidate in rest)
            candidates.extend(rest)
            used.update(rest)
        query_type = WhichQuery if policy.query_kind.value == "which" else CountQuery
        rng.shuffle(candidates)
        return premises, query_type(frozenset({direction}), reference, tuple(candidates)), used
    first, second, *rest = pool
    gold_size = policy.possibility_gold_size
    if len(rest) < gold_size:
        raise PossibilityConstructionError("not enough membership candidates")
    candidate_count = policy.possibility_candidate_count or len(rest)
    if not gold_size < candidate_count <= len(rest):
        raise PossibilityConstructionError(
            "possibility candidate count must exceed gold size and fit the pool"
        )
    candidate_pool = rest[:candidate_count]
    planted, others = candidate_pool[:gold_size], candidate_pool[gold_size:]
    trigger = direction_constraint(first, second, rng.choice(tuple(Direction)))
    any_planted = Or(tuple(member(candidate) for candidate in planted))
    if name == "mp":
        premises = [trigger, Implies(trigger, any_planted)]
    elif name == "iff":
        premises = [trigger, Iff(trigger, any_planted)]
    elif name == "dje":
        if not others:
            raise PossibilityConstructionError("dje needs an excluded candidate")
        premises = [
            Or((*any_planted.operands, member(others[0]))),
            Not(member(others[0])),
        ]
    elif name == "csr":
        cases = _exclusive_cases(first, second, gold_size, rng)
        premises = [
            Or(cases),
            *(
                Implies(case, member(candidate))
                for case, candidate in zip(cases, planted, strict=True)
            ),
        ]
    elif name == "jcr":
        # The planted group moves together, so only counts {0, gold_size} are
        # possible although every planted member is individually contingent.
        premises = [
            Or(
                (
                    And(tuple(member(candidate) for candidate in planted)),
                    And(tuple(Not(member(candidate)) for candidate in planted)),
                )
            ),
        ]
    else:
        raise PossibilityConstructionError(f"unknown skeleton {name}")
    candidates = [*planted, *others]
    premises.extend(
        Not(member(candidate))
        for candidate in others
        if not (name == "dje" and candidate == others[0])
    )
    query_type = WhichQuery if policy.query_kind.value == "which" else CountQuery
    rng.shuffle(candidates)
    used = {reference, first, second, *candidates}
    return premises, query_type(frozenset({direction}), reference, tuple(candidates)), used


def _exclusive_cases(
    first: str, second: str, count: int, rng: Any
) -> tuple[SpatialFormula, ...]:
    """Mutually exclusive cases: distinct exact directions for one pair."""
    return tuple(
        direction_constraint(first, second, value)
        for value in rng.sample(list(Direction), count)
    )


def _exact_chain(
    chain: list[str], direction: Direction
) -> list[SpatialFormula]:
    return [
        direction_constraint(subject, previous, direction)
        for previous, subject in pairwise(chain)
    ]


def _opposite(direction: Direction) -> Direction:
    x_sign, y_sign = direction_signs(direction)
    return _DIRECTION_BY_SIGNS[(-x_sign, -y_sign)]


def _open_axis_chain(chain: list[str], rng: Any) -> list[SpatialFormula]:
    """Fix one axis strictly along the chain and leave the other unordered."""
    fixed_axis = rng.randrange(2)
    fixed_sign = rng.choice((-1, 1))
    free = [1, -1, *[rng.choice((-1, 1)) for _ in range(len(chain) - 3)]]
    rng.shuffle(free)
    relations = []
    for (previous, subject), component in zip(pairwise(chain), free, strict=True):
        signs = (fixed_sign, component) if fixed_axis == 0 else (component, fixed_sign)
        relations.append(
            direction_constraint(subject, previous, _DIRECTION_BY_SIGNS[signs])
        )
    return relations


def _membership_chain(
    chain: list[str], direction: Direction, *, open_axis: bool, rng: Any
) -> list[SpatialFormula]:
    """Reach one diagonal membership; optionally leave one axis unordered."""
    x_sign, y_sign = direction_signs(direction)
    links = len(chain) - 1
    if open_axis:
        open_index = rng.randrange(2)
        mixed = [1, -1, *[rng.choice((-1, 1)) for _ in range(links - 2)]]
        rng.shuffle(mixed)
        signs = [
            (x_sign, component) if open_index == 1 else (component, y_sign)
            for component in mixed
        ]
    else:
        signs = [(x_sign, y_sign)] * links
    return [
        direction_constraint(subject, previous, _DIRECTION_BY_SIGNS[sign])
        for (previous, subject), sign in zip(pairwise(chain), signs, strict=True)
    ]


def _attached_distractors(
    anchors: list[str], spare: list[str], count: int, rng: Any
) -> list[SpatialFormula]:
    """Hang spare objects off the support as a tree.

    Each premise introduces one new leaf, so it can always be satisfied by
    placing that leaf and never constrains the queried objects. The premises
    mention support objects, which makes them look relevant without being so.
    """
    if count > len(spare):
        raise PossibilityConstructionError("not enough spare objects for noise")
    attached = list(anchors)
    premises = []
    for leaf in spare[:count]:
        premises.append(
            direction_constraint(
                leaf, rng.choice(attached), rng.choice(tuple(Direction))
            )
        )
        attached.append(leaf)
    return premises



def _tautologies(
    objects: tuple[str, ...], count: int, rng: Any
) -> list[SpatialFormula]:
    pairs = list(combinations(objects, 2))
    rng.shuffle(pairs)
    fillers = []
    for subject, reference in pairs[:count]:
        atom = direction_constraint(subject, reference, rng.choice(tuple(Direction)))
        fillers.append(Or((atom, Not(atom))))
    return fillers
