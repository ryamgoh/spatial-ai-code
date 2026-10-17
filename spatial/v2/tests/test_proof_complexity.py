from spatial.v2.proof_complexity import (
    measure_certificate_complexity,
    measure_proof_graph,
)
from spatial.v2.proofs import (
    DirectionClaim,
    DirectionProofCertificate,
    ProofRule,
    ProofStep,
)
from spatial.v2.solver import (
    And,
    Direction,
    DirectionQuery,
    Iff,
    Not,
    Or,
    SpatialProblem,
    direction_constraint,
)


def test_proof_graph_measures_dependency_length_not_parallel_size() -> None:
    ab = direction_constraint("A", "B", Direction.EAST)
    bc = direction_constraint("B", "C", Direction.EAST)
    cd = direction_constraint("C", "D", Direction.EAST)
    steps = (
        ProofStep("P1", ProofRule.PREMISE, ab, premise_index=0),
        ProofStep("P2", ProofRule.PREMISE, bc, premise_index=1),
        ProofStep("P3", ProofRule.PREMISE, cd, premise_index=2),
        ProofStep("T1", ProofRule.AXIS_TRANSITIVITY, ab, ("P1", "P2")),
        ProofStep("T2", ProofRule.AXIS_TRANSITIVITY, bc, ("P2", "P3")),
        ProofStep("M", ProofRule.MODUS_PONENS, cd, ("T1", "T2")),
    )

    measured = measure_proof_graph(steps, "M")

    assert measured.semantic_length == 2
    assert measured.semantic_size == 3
    assert measured.critical_signatures == (
        ("axis-transitivity", "modus-ponens"),
    )
    assert measured.ordered_transitions == (
        ("axis-transitivity", "modus-ponens"),
    )


def test_certificate_profile_counts_atoms_connectives_and_proof_metrics() -> None:
    ab = direction_constraint("A", "B", Direction.EAST)
    bc = direction_constraint("B", "C", Direction.EAST)
    ac = direction_constraint("A", "C", Direction.EAST)
    premise = And((Iff(ab, Not(Not(ab))), Or((bc, ac)), bc))
    problem = SpatialProblem(
        ("A", "B", "C"), premise, DirectionQuery("A", "C")
    )
    steps = (
        ProofStep("P1", ProofRule.PREMISE, ab, premise_index=0),
        ProofStep("P2", ProofRule.PREMISE, bc, premise_index=1),
        ProofStep("T", ProofRule.AXIS_TRANSITIVITY, ac, ("P1", "P2")),
        ProofStep(
            "R",
            ProofRule.DIRECTION_RECOMPOSITION,
            DirectionClaim("A", Direction.EAST, "C"),
            ("T",),
        ),
    )
    certificate = DirectionProofCertificate(problem, steps, "R")

    profile = measure_certificate_complexity(problem, certificate)

    assert profile.top_level_premises == 3
    assert profile.atomic_occurrences == 5
    assert profile.unique_atoms == 3
    assert profile.connective_counts == {"iff": 1, "not": 2, "or": 1}
    assert profile.formula_nesting == 4
    assert profile.semantic_length == 1
    assert profile.full_certificate_depth == 2
    assert profile.semantic_proof_size == 1
    assert profile.support_premise_indices == (0, 1)
    assert profile.removable_premises == 1
