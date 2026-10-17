"""Paired satisfiable/inconsistent spatial-cycle diagnostics."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum

from spatial.v2.model_certificates import evaluate_formula_in_model
from spatial.v2.solver import (
    And,
    Direction,
    DirectionQuery,
    RelationConstraint,
    SpatialProblem,
    SpatialSolverV2,
    direction_constraint,
)


class ConsistencyLabel(str, Enum):
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True)
class ConsistencyCertificate:
    objects: tuple[str, ...]
    clean_premises: tuple[RelationConstraint, ...]
    dirty_premises: tuple[RelationConstraint, ...]
    cycle_premise_indices: tuple[int, ...]
    added_premise_index: int
    clean_coordinates: dict[str, tuple[int, int]]
    axis: str


@dataclass(frozen=True)
class ConsistencyPair:
    pair_id: str
    cycle_length: int
    distractor_count: int
    axis: str
    certificate: ConsistencyCertificate

    def rows(self) -> tuple[dict, dict]:
        return (
            _row(self, ConsistencyLabel.CONSISTENT),
            _row(self, ConsistencyLabel.INCONSISTENT),
        )


def _problem(
    objects: tuple[str, ...], premises: tuple[RelationConstraint, ...]
) -> SpatialProblem:
    return SpatialProblem(
        objects,
        And(premises),
        DirectionQuery(objects[0], objects[1]),
    )


def check_consistency_certificate(
    certificate: ConsistencyCertificate,
    *,
    solver: SpatialSolverV2 | None = None,
) -> None:
    solver = solver or SpatialSolverV2()
    if certificate.axis not in {"x", "y"}:
        raise ValueError("consistency certificate axis must be x or y")
    if set(certificate.clean_coordinates) != set(certificate.objects):
        raise ValueError("clean witness must cover every object")
    if any(
        not evaluate_formula_in_model(premise, certificate.clean_coordinates)
        for premise in certificate.clean_premises
    ):
        raise ValueError("clean witness does not satisfy every clean premise")
    if len(certificate.dirty_premises) != len(certificate.clean_premises) + 1:
        raise ValueError("dirty variant must add exactly one premise")
    if not 0 <= certificate.added_premise_index < len(certificate.dirty_premises):
        raise ValueError("added premise index is out of range")
    clean = solver.analyze(_problem(certificate.objects, certificate.clean_premises))
    dirty = solver.analyze(_problem(certificate.objects, certificate.dirty_premises))
    if not clean.consistent or dirty.consistent:
        raise ValueError("certificate does not pair a clean and inconsistent problem")
    cycle = set(certificate.cycle_premise_indices)
    if certificate.added_premise_index not in cycle or len(cycle) < 2:
        raise ValueError("cycle core must contain its planted closing edge")
    for removed in cycle:
        reduced = tuple(
            premise
            for index, premise in enumerate(certificate.dirty_premises)
            if index != removed
        )
        if not solver.analyze(_problem(certificate.objects, reduced)).consistent:
            raise ValueError("planted cycle is not deletion-minimal")


def generate_consistency_pair(
    *,
    names: tuple[str, ...],
    cycle_length: int,
    distractor_count: int,
    axis: str,
    seed: int,
    solver: SpatialSolverV2 | None = None,
) -> ConsistencyPair:
    if not 2 <= cycle_length <= 8:
        raise ValueError("cycle_length must be between 2 and 8")
    if not 0 <= distractor_count <= 8:
        raise ValueError("distractor_count must be between 0 and 8")
    if axis not in {"x", "y"}:
        raise ValueError("axis must be x or y")
    required = cycle_length + 2 * distractor_count
    if len(names) < required:
        raise ValueError(f"consistency pair requires at least {required} names")
    rng = random.Random(seed)
    selected = list(rng.sample(names, required))
    cycle_nodes = selected[:cycle_length]
    distractor_nodes = selected[cycle_length:]
    direction = Direction.EAST if axis == "x" else Direction.NORTH
    clean_core = [
        direction_constraint(first, second, direction)
        for first, second in zip(cycle_nodes, cycle_nodes[1:])
    ]
    distractors = [
        direction_constraint(first, second, direction)
        for first, second in zip(
            distractor_nodes[::2], distractor_nodes[1::2], strict=True
        )
    ]
    clean_premises = tuple((*clean_core, *distractors))
    closing = direction_constraint(cycle_nodes[-1], cycle_nodes[0], direction)
    dirty_premises = tuple((*clean_premises, closing))
    objects = tuple(sorted(selected))
    solver = solver or SpatialSolverV2()
    clean_analysis = solver.analyze(_problem(objects, clean_premises))
    if not clean_analysis.consistent or clean_analysis.coordinates is None:
        raise ValueError("failed to construct a satisfiable clean cycle base")
    added_index = len(dirty_premises) - 1
    certificate = ConsistencyCertificate(
        objects=objects,
        clean_premises=clean_premises,
        dirty_premises=dirty_premises,
        cycle_premise_indices=tuple((*range(len(clean_core)), added_index)),
        added_premise_index=added_index,
        clean_coordinates=clean_analysis.coordinates,
        axis=axis,
    )
    check_consistency_certificate(certificate, solver=solver)
    identity = hashlib.sha256(
        f"{seed}:{cycle_length}:{distractor_count}:{axis}".encode()
    ).hexdigest()[:16]
    return ConsistencyPair(
        pair_id=f"spatial-consistency-{identity}",
        cycle_length=cycle_length,
        distractor_count=distractor_count,
        axis=axis,
        certificate=certificate,
    )


def _render_premise(premise: RelationConstraint) -> str:
    direction = next(iter(premise.allowed)).value
    return f"{premise.subject} is {direction} of {premise.reference}."


def _row(pair: ConsistencyPair, label: ConsistencyLabel) -> dict:
    premises = (
        pair.certificate.clean_premises
        if label is ConsistencyLabel.CONSISTENT
        else pair.certificate.dirty_premises
    )
    text = " ".join(_render_premise(premise) for premise in premises)
    answer = "A" if label is ConsistencyLabel.CONSISTENT else "B"
    return {
        "id": f"{pair.pair_id}-{label.value}",
        "messages": [
            {
                "role": "system",
                "content": "Determine whether all stated spatial constraints can hold at once.",
            },
            {
                "role": "user",
                "content": (
                    f"{text}\n\nAre all of these spatial statements mutually "
                    "consistent? A. Consistent, B. Inconsistent"
                ),
            },
            {"role": "assistant", "content": f"Answer: {answer}"},
        ],
        "metadata": {
            "schema": "spatial-consistency-v1",
            "task_kind": "consistency",
            "pair_id": pair.pair_id,
            "label": label.value,
            "cycle_length": pair.cycle_length,
            "unsat_core_size": pair.cycle_length,
            "contradiction_proof_depth": pair.cycle_length,
            "axis": pair.axis,
            "distractor_count": pair.distractor_count,
            "oracle_letters": [answer],
        },
    }
