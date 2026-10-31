"""Mutation checks for the independently implemented artifact auditor."""

import importlib.util
from pathlib import Path

import pytest

from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_evidence import build_evidence, render_pair
from spatial.v2.tests.test_entailment_positions import fixture_record
from spatial.v2.tests.test_pivot_evidence import BY_ID

PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments/spatialentail-composition-v1/audit_entailment_candidate.py"
)
SPEC = importlib.util.spec_from_file_location("independent_entailment_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def record(name="S3", menu=("U", "V", "W", "T")):
    value = fixture_record(name, menu)
    evidence = build_evidence(BY_ID[name].problem())
    solver, claims = AUDIT.independent_claims(evidence.semantics.problem)
    categories = {
        v: "contingent" if AUDIT.checked_sat(solver, claim) else "impossible"
        for v, claim in claims.items()
        if evidence.semantics.consistent and v not in evidence.semantics.entailed
    }
    for row in value["rows"]:
        row["negative_menu_categories"] = {
            letter: categories[v] for letter, v in zip("ABCD", menu) if v in categories
        }
        row["references"] = render_pair(
            evidence, tuple(row["menu"]), Contract(row["contract"])
        )
    return value


def test_independent_auditor_accepts_verified_fixture():
    assert AUDIT.audit_record(record()) == (2, 2)


@pytest.mark.parametrize(
    ("name", "menu", "counts"),
    [
        ("C1", (0, 1, 2, 3), (0, 5)),
        ("C2", (0, 1, 2, 3), (1, 4)),
        ("C3", (0, 1, 2, 3), (1, 4)),
        ("I1", ("North", "East", "South", "West"), (0, 0)),
        ("I3", ("North", "East", "South", "West"), (0, 0)),
        ("D1", ("North", "East", "South", "West"), (1, 7)),
        ("S2", ("U", "V", "W", "T"), (0, 4)),
        ("S4", ("U", "V", "W", "T"), (0, 4)),
    ],
)
def test_independent_auditor_covers_query_and_escape_categories(name, menu, counts):
    assert AUDIT.audit_record(record(name, menu)) == counts


def test_independent_auditor_rejects_altered_conflict():
    value = record("I1", ("North", "East", "South", "West"))
    value["evidence"]["conflict"]["paths"][0][0]["strict"] = False
    with pytest.raises(ValueError, match="conflict strictness"):
        AUDIT.audit_record(value)


def test_independent_auditor_rejects_altered_gold():
    value = record()
    value["rows"][1]["answer"] = "F"
    with pytest.raises(ValueError, match="gold mismatch"):
        AUDIT.audit_record(value)


def test_independent_auditor_rejects_altered_proof():
    value = record()
    support = next(iter(value["evidence"]["positive_support"].values()))
    support["paths"][0]["signs"][0] = 1
    with pytest.raises(ValueError, match="proof signs"):
        AUDIT.audit_record(value)


def test_independent_auditor_rejects_incomplete_counterexample():
    value = record()
    points = next(iter(value["evidence"]["semantics"]["counterexamples"].values()))
    points.pop(next(iter(points)))
    with pytest.raises(ValueError, match="incomplete counterexample"):
        AUDIT.audit_record(value)


def test_independent_contract_truth_table():
    assert AUDIT.label(True, {"U", "V"}, ("U", "W", "T", "X"), "unique-entailed") == "E"
    assert AUDIT.label(True, {"U", "V"}, ("U", "W", "T", "X"), "listed-entailed") == "A"
    assert (
        AUDIT.label(True, {"U", "V"}, ("U", "W", "T", "X"), "exhaustive-entailed")
        == "F"
    )
    assert AUDIT.label(True, set(), ("U", "W", "T", "X"), "listed-entailed") == "F"
