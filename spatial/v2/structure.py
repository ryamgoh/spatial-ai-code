"""Canonical visible-problem structure for conservative dataset grouping.

Entity names, object/candidate order, and commutative operand order are ignored.
Directions and implication order remain content. Refinement is invariant but
not a complete graph-isomorphism test; large unresolved symmetry classes use
a coarse key that can merge distinct structures rather than split renamed ones.
"""

from __future__ import annotations

import json
from functools import lru_cache
from itertools import permutations, product
from math import factorial, prod

from spatial.v2.solver import (
    And,
    DirectionQuery,
    Iff,
    Implies,
    Not,
    Or,
    RelationConstraint,
    SpatialFormula,
    SpatialProblem,
)


def _encode(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _formula(formula: SpatialFormula, names: dict[str, int]) -> tuple:
    if isinstance(formula, RelationConstraint):
        return (
            "relation",
            names[formula.subject],
            names[formula.reference],
            tuple(sorted(direction.name for direction in formula.allowed)),
        )
    if isinstance(formula, Not):
        return ("not", _formula(formula.operand, names))
    if isinstance(formula, (And, Or)):
        return (
            type(formula).__name__,
            tuple(
                sorted(
                    (_formula(child, names) for child in formula.operands), key=_encode
                )
            ),
        )
    if isinstance(formula, Iff):
        return (
            "iff",
            tuple(
                sorted(
                    (_formula(formula.left, names), _formula(formula.right, names)),
                    key=_encode,
                )
            ),
        )
    if isinstance(formula, Implies):
        return (
            "implies",
            _formula(formula.antecedent, names),
            _formula(formula.consequent, names),
        )
    raise TypeError(f"unsupported formula: {type(formula).__name__}")


def _problem(problem: SpatialProblem, names: dict[str, int]) -> tuple:
    query = problem.query
    if isinstance(query, DirectionQuery):
        query_key = (
            "direction",
            names[query.target],
            names[query.reference],
            tuple(sorted(direction.name for direction in query.candidate_directions)),
        )
    else:
        query_key = (
            type(query).__name__,
            names[query.reference],
            tuple(sorted(names[name] for name in query.candidates)),
            tuple(sorted(direction.name for direction in query.directions)),
        )
    return (tuple(sorted(names.values())), _formula(problem.premise, names), query_key)


@lru_cache(maxsize=2048)
def canonical_structure_profile(
    problem: SpatialProblem,
    *,
    max_permutations: int = 40_320,
) -> dict[str, object]:
    """Return an invariant key, exact when residual permutations fit the budget.

    Exact means syntactic formula/query isomorphism under the documented
    permutations, not logical equivalence, direction rotation, or proof identity.
    The budget must be fixed across a workload so fallback grouping is uniform.
    """
    if max_permutations < 1:
        raise ValueError("max_permutations must be positive")
    colors = dict.fromkeys(problem.objects, 0)
    for _ in problem.objects:
        descriptors = {
            name: _encode(
                (
                    colors[name],
                    _problem(
                        problem,
                        {
                            other: -1 if other == name else colors[other]
                            for other in problem.objects
                        },
                    ),
                )
            )
            for name in problem.objects
        }
        palette = {
            value: index
            for index, value in enumerate(sorted(set(descriptors.values())))
        }
        refined = {name: palette[value] for name, value in descriptors.items()}
        stable = len(set(refined.values())) == len(set(colors.values()))
        colors = refined
        if stable:
            break
    groups = [
        tuple(name for name in problem.objects if colors[name] == color)
        for color in sorted(set(colors.values()))
    ]
    permutation_count = prod(factorial(len(group)) for group in groups)
    exact = permutation_count <= max_permutations
    if exact:
        canonical = min(
            _encode(
                _problem(
                    problem,
                    {
                        name: index
                        for index, name in enumerate(
                            name for group in ordered_groups for name in group
                        )
                    },
                )
            )
            for ordered_groups in product(*(permutations(group) for group in groups))
        )
    else:
        canonical = _encode(_problem(problem, colors))
    return {
        "schema": "visible-problem-structure-v1",
        "canonicalization": "exact" if exact else "coarse-refinement",
        "permutation_budget": max_permutations,
        "direction_sensitive": True,
        "canonical_problem": canonical,
    }
