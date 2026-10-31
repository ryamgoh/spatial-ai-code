"""Reconstruct saved typed evidence for explicitly versioned artifact revision."""

from spatial.v2.composition_conflicts import Conflict, Edge
from spatial.v2.entailment_contracts import VERSION, Semantics
from spatial.v2.entailment_evidence import Evidence
from spatial.v2.entailment_scene_protocol import parse_row
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import CountQuery


def load_evidence(record):
    if record.get("interface") != VERSION or not record.get("rows"):
        raise ValueError("saved record must contain entailment-only views")
    first = record["rows"][0]
    protocol = (first.get("prompt_version"), first.get("scope_policy"))
    if any(
        (row.get("prompt_version"), row.get("scope_policy")) != protocol
        for row in record["rows"]
    ):
        raise ValueError("saved record mixes scene protocols")
    if (record.get("prompt_version"), record.get("scope_policy")) != protocol:
        raise ValueError("saved record and views disagree on scene protocol")
    problem, _, _ = parse_row(first)
    if record["problem"] != tagged_dataclass_to_dict(problem):
        raise ValueError("saved problem differs from the parsed prompt")
    saved = record["evidence"]
    if saved["type"] != "Evidence":
        raise ValueError("unknown saved evidence schema")
    sem = saved["semantics"]
    value = int if isinstance(problem.query, CountQuery) else str
    semantics = Semantics(
        problem=problem,
        consistent=sem["consistent"],
        entailed=frozenset(value(v) for v in sem["entailed"]),
        counterexamples={
            value(v): {name: tuple(point) for name, point in coordinates.items()}
            for v, coordinates in sem["counterexamples"].items()
        },
        engine=sem["engine"],
    )
    conflict = saved["conflict"]
    if conflict is not None:
        conflict = Conflict(
            problem,
            conflict["kind"],
            tuple(tuple(Edge(**edge) for edge in path) for path in conflict["paths"]),
        )
    return Evidence(
        semantics,
        {value(v): support for v, support in saved["positive_support"].items()},
        conflict,
    )
