"""Independent selected-certificate depth checks on fully replayed Unique records."""

from copy import deepcopy

from audit_entailment_candidate import audit_record, require

from spatial.v2.entailment_dataset import audit_row
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.entailment_scene_protocol import parse_row
from spatial.v2.solver import CountQuery, Direction, DirectionQuery


def certificate_depth(support):
    paths = (
        [path for member in support["memberships"] for path in member["paths"]]
        if support["kind"] == "fixed-count"
        else support["paths"]
    )
    require(bool(paths), "empty selected certificate")
    return max(len(path["indices"]) for path in paths)


def replay(record):
    """Return an annotated copy; never promote compact metadata without evidence."""
    audit_record(record)
    evidence = load_evidence(record)
    result = deepcopy(record)
    problem = evidence.semantics.problem
    # Reconstruct the renderer's complete-domain ordering independently.
    domain = (
        [d.value for d in Direction]
        if isinstance(problem.query, DirectionQuery)
        else list(range(len(problem.query.candidates) + 1))
        if isinstance(problem.query, CountQuery)
        else list(problem.query.candidates)
    )
    positives = evidence.positive_support
    values = [v for v in domain if v in positives]
    for row in result["rows"]:
        require(row["contract"] == "unique-entailed", "retired contract in depth audit")
        _, menu, _ = parse_row(row)
        actual = None
        if evidence.semantics.consistent and len(values) == 1:
            actual = certificate_depth(positives[values[0]])
            expected = "ABCD"[menu.index(values[0])] if values[0] in menu else "F"
            require(row["answer"] == expected, "sole answer/menu mismatch")
        elif evidence.semantics.consistent and len(values) > 1:
            cheapest = sorted(certificate_depth(s) for s in positives.values())[1]
            emitted = max(certificate_depth(positives[v]) for v in values[:2])
            if emitted == cheapest:
                actual = emitted
        if "selected_explanation_depth" in row:
            require(
                row["selected_explanation_depth"] == actual, "selected depth mismatch"
            )
        row["selected_explanation_depth"] = actual
        row["selected_explanation_depth_support"] = (
            actual is not None and actual == row["depth"]
        )
        # Verify that the visible reference text is exactly the evidence-derived
        # renderer output, not merely a correct final answer.
        audit_row(row, evidence)
        row["selected_depth_audit"] = "passed"
    result["independent_audit"] = "passed"
    return result
