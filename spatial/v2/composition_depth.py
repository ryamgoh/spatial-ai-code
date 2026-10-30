"""Calculus-relative depth descriptors, not minimum semantic reasoning depth."""

from dataclasses import replace

from spatial.v2.composition_contracts import UnsupportedEvidence, analyze
from spatial.v2.difficulty import Axis, _axis_path, derive_axis_proof
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import (
    And,
    Direction,
    DirectionQuery,
    SpatialSolverV2,
    WhichQuery,
    direction_signs,
)

VERSION = "shortest-atomic-axis-support-v1"


def signature(semantics):
    return semantics.consistent, semantics.possible, semantics.certain


def _record(paths):
    return {
        "depth": max(len(path.premise_indices) for path in paths),
        "paths": [tagged_dataclass_to_dict(path) for path in paths],
        "premise_indices": sorted(
            {index for path in paths for index in path.premise_indices}
        ),
    }


def membership_support(problem, candidate, qualifies):
    """Shortest positive pair, or shortest single-axis exclusion certificate."""
    query = problem.query
    direction = next(iter(query.directions))
    if qualifies:
        proof = derive_axis_proof(problem, candidate, query.reference, direction)
        if proof is None:
            raise UnsupportedEvidence("positive membership lacks axis support")
        result = _record((proof.x, proof.y))
    else:
        paths = []
        for axis, required in zip((Axis.X, Axis.Y), direction_signs(direction)):
            for sign in (-1, 0, 1):
                if sign == required:
                    continue
                path = _axis_path(problem, axis, candidate, query.reference, sign)
                if path is not None:
                    paths.append(path)
        if not paths:
            raise UnsupportedEvidence("negative membership lacks axis exclusion")
        path = min(
            paths,
            key=lambda p: (
                len(p.premise_indices),
                p.axis.value,
                p.relation.value,
                p.path,
            ),
        )
        result = _record((path,))
    return {"candidate": candidate, "qualifies": qualifies, **result}


def measure_depth(problem, *, solver=None):
    """Verify full semantics; refuse inapplicable or unsupported measurements.

    `_axis_path` is the shared BFS used by derive_axis_proof. Reuse its strict
    and equality rules for negative membership rather than duplicating them.
    """
    solver = solver or SpatialSolverV2(backend="z3", timeout_ms=5000)
    semantics = analyze(problem, solver=solver, include_witnesses=False)
    if not semantics.consistent:
        raise UnsupportedEvidence("inconsistent: no entailed-depth descriptor")
    query = problem.query
    if isinstance(query, DirectionQuery):
        if len(semantics.possible) != 1:
            raise UnsupportedEvidence("ambiguous-direction")
        proof = derive_axis_proof(
            problem,
            query.target,
            query.reference,
            Direction(next(iter(semantics.possible))),
        )
        if proof is None:
            raise UnsupportedEvidence("direction lacks complete axis support")
        return {"version": VERSION, **_record((proof.x, proof.y))}

    membership_problem = replace(
        problem, query=WhichQuery(query.directions, query.reference, query.candidates)
    )
    memberships = analyze(membership_problem, solver=solver, include_witnesses=False)
    if memberships.possible != memberships.certain:
        raise UnsupportedEvidence("non-fixed-memberships")
    supports = [
        membership_support(problem, candidate, candidate in memberships.certain)
        for candidate in query.candidates
    ]
    if not supports:
        raise UnsupportedEvidence("empty-candidate-domain")
    depth = max(item["depth"] for item in supports)
    ablations = []
    for item in supports:
        if item["depth"] != depth:
            continue
        for index in item["premise_indices"]:
            retained = tuple(
                atom for i, atom in enumerate(problem.premise.operands) if i != index
            )
            if not retained:
                # The atomic interface does not represent an empty conjunction.
                continue
            reduced = replace(problem, premise=And(retained))
            after = analyze(reduced, solver=solver, include_witnesses=False)
            changed = signature(after) != signature(semantics)
            ablations.append(
                {
                    "candidate": item["candidate"],
                    "removed_premise_index": index,
                    "possible_after": sorted(after.possible),
                    "certain_after": sorted(after.certain),
                    "answer_changed": changed,
                }
            )
            if changed:
                return {
                    "version": VERSION,
                    "depth": depth,
                    "memberships": supports,
                    "ablations": ablations,
                    "dependence_check": "passed-not-a-minimality-proof",
                }
    raise UnsupportedEvidence(
        "maximum-depth-support-has-no-demonstrated-answer-dependence"
    )
