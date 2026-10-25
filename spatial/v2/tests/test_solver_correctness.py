"""Independent finite-model checks for the qualitative correctness argument."""

import random
from dataclasses import replace
from functools import cache
from itertools import permutations, product

import pytest

from spatial.v2.grading import AnswerMode, ResolutionStatus, resolve_answer
from spatial.v2.model_certificates import evaluate_formula_in_model
from spatial.v2.proofs import (
    AxisFact,
    Contradiction,
    OrderRelation,
    ProofAxis,
    ProofCheckError,
    ProofConstructionError,
    ProofRule,
    ProofStep,
    _check_step,
    build_direction_proof,
    build_formula_refutation,
    check_direction_proof,
    check_formula_refutation,
)
from spatial.v2.solver import (
    And,
    CountAnalysis,
    CountQuery,
    Direction,
    DirectionAnalysis,
    DirectionQuery,
    Iff,
    Implies,
    Not,
    Or,
    RelationConstraint,
    SpatialProblem,
    SpatialSolverV2,
    WhichAnalysis,
    WhichQuery,
    _compose_directions,
    _propagate,
    _required_atoms,
    _Z3Engine,
)
from spatial.v2.text import SpatialTextAdapter, render_spatial_formula

OBJECTS = ("A", "B", "C")
EASTWARD = frozenset({Direction.NORTHEAST, Direction.EAST, Direction.SOUTHEAST})


@cache
def ranked_worlds(objects):
    # Every qualitative n-object world has a rank-equivalent world on this grid.
    points = tuple(product(range(len(objects)), repeat=2))
    return tuple(
        {
            name: (point[0], point[1])
            for name, point in zip(objects, assignment, strict=True)
        }
        for assignment in permutations(points, len(objects))
    )


def actual_direction(subject, reference):
    """Do not reuse the production sign table, composition, or evaluator."""
    x, y = subject
    other_x, other_y = reference
    if x == other_x:
        assert y != other_y
        return Direction.NORTH if y > other_y else Direction.SOUTH
    if y == other_y:
        return Direction.EAST if x > other_x else Direction.WEST
    if x > other_x:
        return Direction.NORTHEAST if y > other_y else Direction.SOUTHEAST
    return Direction.NORTHWEST if y > other_y else Direction.SOUTHWEST


def holds(formula, world):
    if isinstance(formula, RelationConstraint):
        return (
            actual_direction(world[formula.subject], world[formula.reference])
            in formula.allowed
        )
    if isinstance(formula, Not):
        return not holds(formula.operand, world)
    if isinstance(formula, And):
        return all(holds(operand, world) for operand in formula.operands)
    if isinstance(formula, Or):
        return any(holds(operand, world) for operand in formula.operands)
    if isinstance(formula, Implies):
        return not holds(formula.antecedent, world) or holds(formula.consequent, world)
    if isinstance(formula, Iff):
        return holds(formula.left, world) == holds(formula.right, world)
    raise AssertionError(f"unsupported oracle formula: {formula!r}")


def atom(subject, reference, direction=Direction.NORTHEAST):
    return RelationConstraint(subject, reference, frozenset({direction}))


def formula_cases():
    p, q, r = atom("A", "B"), atom("C", "B"), atom("A", "C")
    cases = [
        ("atom", p),
        ("all-directions", RelationConstraint("A", "B", frozenset(Direction))),
        ("coarse", RelationConstraint("A", "B", EASTWARD)),
        ("not-diagonal", Not(p)),
        ("not-cardinal", Not(atom("A", "B", Direction.EAST))),
        ("and", And((p, q))),
        ("or", Or((p, q))),
        ("implication", Implies(p, q)),
        ("equivalence", Iff(p, q)),
        ("tautology", Or((p, Not(p)))),
        ("contradiction", And((p, Not(p)))),
        ("modus-ponens", And((p, Implies(p, q)))),
        ("vacuous-implication", And((Not(p), Implies(p, q)))),
        ("equivalence-negative", And((Iff(p, q), Not(p)))),
        ("de-morgan", Not(And((p, q)))),
        ("case-split", And((Or((p, q)), Implies(p, r), Implies(q, r)))),
        ("disjunctive-syllogism", And((Or((p, q)), Not(p)))),
        ("correlated-xor", And((Or((p, q)), Not(And((p, q)))))),
        (
            "axis-equality",
            And((atom("A", "B", Direction.NORTH), atom("B", "C", Direction.EAST))),
        ),
        (
            "strict-cycle",
            And(
                tuple(
                    RelationConstraint(first, second, EASTWARD)
                    for first, second in (("A", "B"), ("B", "C"), ("C", "A"))
                )
            ),
        ),
    ]
    rng = random.Random(20261021)

    def sampled(depth):
        if depth == 0 or rng.random() < 0.3:
            subject, reference = rng.sample(OBJECTS, 2)
            domain = frozenset(rng.sample(tuple(Direction), rng.randint(1, 8)))
            return RelationConstraint(subject, reference, domain)
        operator = rng.choice((Not, And, Or, Implies, Iff))
        if operator is Not:
            return Not(sampled(depth - 1))
        left, right = sampled(depth - 1), sampled(depth - 1)
        if operator is And:
            return And((left, right))
        if operator is Or:
            return Or((left, right))
        if operator is Implies:
            return Implies(left, right)
        return Iff(left, right)

    cases.extend((f"nested-{index}", sampled(3)) for index in range(20))
    return tuple(cases)


FORMULAS = formula_cases()
QUERIES = (
    DirectionQuery("A", "B"),
    DirectionQuery("A", "B", frozenset({Direction.NORTHEAST, Direction.NORTHWEST})),
    WhichQuery(frozenset({Direction.NORTHEAST}), "B", ("A", "C")),
    WhichQuery(EASTWARD, "B", ("A", "C")),
    CountQuery(frozenset({Direction.NORTHEAST}), "B", ("A", "C")),
    CountQuery(EASTWARD, "B", ("A", "C")),
)


def members(query, world):
    return tuple(
        candidate
        for candidate in query.candidates
        if actual_direction(world[candidate], world[query.reference])
        in query.directions
    )


def validate_world(world, problem):
    assert set(world) == set(problem.objects)
    assert all(type(value) is int for point in world.values() for value in point)
    assert len(set(world.values())) == len(world)
    assert holds(problem.premise, world)
    assert all(
        0 <= value < len(problem.objects) for point in world.values() for value in point
    )


@pytest.mark.parametrize("backend", ("reference", "z3"))
@pytest.mark.parametrize(
    "query",
    QUERIES,
    ids=("dir", "restricted", "which", "which-coarse", "count", "count-coarse"),
)
@pytest.mark.parametrize("name,premise", FORMULAS, ids=[name for name, _ in FORMULAS])
def test_query_answers_match_all_ranked_worlds(backend, query, name, premise):
    if backend == "z3":
        pytest.importorskip("z3")
    problem = SpatialProblem(OBJECTS, premise, query)
    worlds = tuple(world for world in ranked_worlds(OBJECTS) if holds(premise, world))
    solver = SpatialSolverV2(backend=backend)
    analysis = solver.analyze(problem)
    assert analysis.error is None, name
    assert analysis.consistent is bool(worlds), name
    assert solver.analyze(problem, include_witnesses=False).consistent is bool(worlds)
    if isinstance(query, DirectionQuery):
        assert isinstance(analysis, DirectionAnalysis)
        expected = {
            actual_direction(world[query.target], world[query.reference])
            for world in worlds
        } & query.candidate_directions
        assert set(analysis.possible_directions) == expected
        assert solver.semantic_signature(problem) == (
            (True, analysis.possible_directions) if worlds else (False,)
        )
    else:
        possible = tuple(
            candidate
            for candidate in query.candidates
            if any(candidate in members(query, world) for world in worlds)
        )
        entailed = tuple(
            candidate
            for candidate in query.candidates
            if worlds and all(candidate in members(query, world) for world in worlds)
        )
        if isinstance(query, WhichQuery):
            assert isinstance(analysis, WhichAnalysis)
            assert analysis.possible_entities == possible
            assert analysis.entailed_entities == entailed
            signature = (True, possible, entailed)
        else:
            assert isinstance(analysis, CountAnalysis)
            expected_counts = tuple(
                sorted({len(members(query, world)) for world in worlds})
            )
            assert analysis.possible_counts == expected_counts
            signature = (True, expected_counts, possible, entailed)
        assert solver.semantic_signature(problem) == (signature if worlds else (False,))
    for answer, world in analysis.witnesses.items():
        validate_world(world, problem)
        if isinstance(query, DirectionQuery):
            assert (
                actual_direction(world[query.target], world[query.reference]) == answer
            )
        elif isinstance(query, WhichQuery):
            assert answer in members(query, world)
        else:
            assert len(members(query, world)) == answer


@pytest.mark.parametrize("backend", ("reference", "z3"))
@pytest.mark.parametrize("name,premise", FORMULAS, ids=[name for name, _ in FORMULAS])
def test_arbitrary_claim_assessment_matches_all_ranked_worlds(backend, name, premise):
    if backend == "z3":
        pytest.importorskip("z3")
    problem = SpatialProblem(OBJECTS, premise, DirectionQuery("A", "B"))
    claim = Implies(atom("A", "B"), Not(atom("C", "B")))
    worlds = tuple(world for world in ranked_worlds(OBJECTS) if holds(premise, world))
    assessment = SpatialSolverV2(backend=backend).assess(problem, claim)
    assert assessment.error is None, name
    assert assessment.consistent is bool(worlds)
    assert assessment.possible is any(holds(claim, world) for world in worlds)
    assert assessment.entailed is (
        bool(worlds) and all(holds(claim, world) for world in worlds)
    )
    for world, expected in (
        (assessment.witness, True),
        (assessment.counterexample, False),
    ):
        if world is not None:
            validate_world(world, problem)
            assert holds(claim, world) is expected


def test_every_atomic_domain_and_complement_match_independent_coordinates():
    z3 = pytest.importorskip("z3")
    engine = _Z3Engine(timeout_ms=5_000)
    objects = ("A", "B")
    x = {name: z3.Int(f"x_{name}") for name in objects}
    y = {name: z3.Int(f"y_{name}") for name in objects}
    for mask in range(1, 256):
        domain = frozenset(
            direction
            for index, direction in enumerate(Direction)
            if mask & (1 << index)
        )
        formula = RelationConstraint("A", "B", domain)
        encoded = engine._formula(x, y, formula)
        for world in ranked_worlds(objects):
            replacements = tuple(
                replacement
                for name in objects
                for replacement in (
                    (x[name], z3.IntVal(world[name][0])),
                    (y[name], z3.IntVal(world[name][1])),
                )
            )
            value = z3.simplify(z3.substitute(encoded, *replacements))
            assert z3.is_true(value) is holds(formula, world)
            assert z3.is_true(z3.simplify(z3.Not(value))) is holds(Not(formula), world)


def test_composition_table_is_exact_for_every_three_object_order_type():
    observed = {(first, second): set() for first in Direction for second in Direction}
    for world in ranked_worlds(OBJECTS):
        first = actual_direction(world["A"], world["B"])
        second = actual_direction(world["B"], world["C"])
        observed[first, second].add(actual_direction(world["A"], world["C"]))
    for (first, second), expected in observed.items():
        assert _compose_directions(frozenset({first}), frozenset({second})) == expected


@pytest.mark.parametrize("name,premise", FORMULAS, ids=[name for name, _ in FORMULAS])
def test_preprocessing_never_discards_a_valid_boolean_world(name, premise):
    required = _required_atoms(premise)
    propagated = _propagate(OBJECTS, required)
    for world in ranked_worlds(OBJECTS):
        if holds(premise, world):
            assert all(holds(constraint, world) for constraint in required), name
            assert propagated is not None, name
            assert all(holds(constraint, world) for constraint in propagated), name


def test_preprocessing_preserves_models_for_seeded_relation_networks():
    rng = random.Random(20261022)

    def constraint():
        subject, reference = rng.sample(OBJECTS, 2)
        return RelationConstraint(
            subject,
            reference,
            frozenset(rng.sample(tuple(Direction), rng.randint(1, 8))),
        )

    for _ in range(100):
        constraints = tuple(constraint() for _ in range(4))
        premise = And(constraints)
        propagated = _propagate(OBJECTS, constraints)
        for world in ranked_worlds(OBJECTS):
            if holds(premise, world):
                assert propagated is not None
                assert all(holds(constraint, world) for constraint in propagated)


def axis_holds(fact, world):
    axis = 0 if fact.axis is ProofAxis.X else 1
    left, right = world[fact.subject][axis], world[fact.reference][axis]
    return {
        OrderRelation.LESS: left < right,
        OrderRelation.EQUAL: left == right,
        OrderRelation.GREATER: left > right,
    }[fact.relation]


@pytest.mark.parametrize("axis", tuple(ProofAxis))
def test_axis_transitivity_checker_accepts_exactly_the_valid_order_rules(axis):
    for first, second, conclusion in product(OrderRelation, repeat=3):
        left = AxisFact(axis, "A", first, "B")
        right = AxisFact(axis, "B", second, "C")
        result = AxisFact(axis, "A", conclusion, "C")
        expected = all(
            axis_holds(result, world)
            for world in ranked_worlds(OBJECTS)
            if axis_holds(left, world) and axis_holds(right, world)
        )
        previous = {
            "L": ProofStep("L", ProofRule.PREMISE, left),
            "R": ProofStep("R", ProofRule.PREMISE, right),
        }
        step = ProofStep("S", ProofRule.AXIS_TRANSITIVITY, result, ("L", "R"))
        if expected:
            _check_step(step, previous, ())
        else:
            with pytest.raises(ProofCheckError):
                _check_step(step, previous, ())


def boolean_rule_cases():
    p, q = atom("A", "B"), atom("C", "B")
    return (
        (ProofRule.AND_ELIMINATION, (And((p, q)),), p),
        (ProofRule.AND_INTRODUCTION, (p, q), And((p, q))),
        (ProofRule.MODUS_PONENS, (Implies(p, q), p), q),
        (ProofRule.DISJUNCTIVE_SYLLOGISM, (Or((p, q)), Not(p)), q),
        (ProofRule.IFF_ELIMINATION, (Iff(p, q), p), q),
        (ProofRule.IFF_ELIMINATION, (Iff(p, q), q), p),
        (ProofRule.DOUBLE_NEGATION, (Not(Not(p)),), p),
        (ProofRule.CONTRADICTION, (p, Not(p)), Contradiction()),
    )


@pytest.mark.parametrize(
    "rule,inputs,conclusion",
    boolean_rule_cases(),
    ids=[
        f"{rule.value}-{index}"
        for index, (rule, _, _) in enumerate(boolean_rule_cases())
    ],
)
def test_boolean_checker_rules_preserve_truth_in_all_ranked_worlds(
    rule, inputs, conclusion
):
    previous = {
        f"P{index}": ProofStep(
            f"P{index}", ProofRule.PREMISE, formula, premise_index=index
        )
        for index, formula in enumerate(inputs)
    }
    step = ProofStep("S", rule, conclusion, tuple(previous))
    _check_step(step, previous, inputs)
    for world in ranked_worlds(OBJECTS):
        if all(holds(formula, world) for formula in inputs):
            assert not isinstance(conclusion, Contradiction)
            assert holds(conclusion, world)


@pytest.mark.parametrize(
    "name", ("modus-ponens", "disjunctive-syllogism", "case-split", "axis-equality")
)
def test_accepted_proofs_and_refutations_agree_with_independent_worlds(name):
    premise = dict(FORMULAS)[name]
    worlds = tuple(world for world in ranked_worlds(OBJECTS) if holds(premise, world))
    pair = ("A", "C") if name in {"case-split", "axis-equality"} else ("C", "B")
    problem = SpatialProblem(OBJECTS, premise, DirectionQuery(*pair))
    proof = build_direction_proof(problem)
    check_direction_proof(proof)
    claim = atom(*pair, proof.conclusion.direction)
    assert worlds
    assert all(holds(claim, world) for world in worlds)
    if name != "axis-equality":
        refutation = build_formula_refutation(problem, Not(claim))
        check_formula_refutation(refutation)
        assert not any(holds(refutation.claim, world) for world in worlds)
    forged = replace(
        proof.steps[-1],
        conclusion=replace(proof.conclusion, direction=Direction.SOUTH),
    )
    with pytest.raises(ProofCheckError):
        check_direction_proof(replace(proof, steps=(*proof.steps[:-1], forged)))


def test_refutation_constructor_fails_closed_when_a_derivation_is_missing():
    pytest.importorskip("z3")
    premise = dict(FORMULAS)["axis-equality"]
    problem = SpatialProblem(OBJECTS, premise, DirectionQuery("A", "C"))
    claim = atom("A", "C")
    assert SpatialSolverV2(backend="z3").assess(problem, claim).entailed
    with pytest.raises(ProofConstructionError, match="do not refute"):
        build_formula_refutation(problem, Not(claim))


def renderable(formula):
    # Arbitrary structured direction sets need an explicit disjunction in text.
    if isinstance(formula, RelationConstraint):
        atoms = tuple(
            atom(formula.subject, formula.reference, direction)
            for direction in Direction
            if direction in formula.allowed
        )
        return atoms[0] if len(atoms) == 1 else Or(atoms)
    if isinstance(formula, Not):
        return Not(renderable(formula.operand))
    if isinstance(formula, And):
        return And(tuple(renderable(operand) for operand in formula.operands))
    if isinstance(formula, Or):
        return Or(tuple(renderable(operand) for operand in formula.operands))
    if isinstance(formula, Implies):
        return Implies(renderable(formula.antecedent), renderable(formula.consequent))
    if isinstance(formula, Iff):
        return Iff(renderable(formula.left), renderable(formula.right))
    raise AssertionError(f"unsupported renderable formula: {formula!r}")


@pytest.mark.parametrize("name,premise", FORMULAS, ids=[name for name, _ in FORMULAS])
def test_model_checker_and_text_round_trip_match_independent_evaluation(name, premise):
    text = (
        "Consider a map with multiple objects:\n"
        "A is in the map. B is in the map. C is in the map. "
        f"{render_spatial_formula(renderable(premise))}.\n\n"
        "Question: In which direction is A relative to B?"
    )
    parsed = SpatialTextAdapter().parse(text)
    for world in ranked_worlds(OBJECTS):
        expected = holds(premise, world)
        assert holds(parsed.problem.premise, world) is expected, name
        assert evaluate_formula_in_model(premise, world) is expected, name


@pytest.mark.parametrize("query", QUERIES)
def test_unknown_is_an_error_not_an_impossibility_or_entailment(monkeypatch, query):
    z3 = pytest.importorskip("z3")
    solver = SpatialSolverV2(backend="z3")
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    monkeypatch.setattr(z3.Solver, "reason_unknown", lambda self: "injected unknown")
    problem = SpatialProblem(OBJECTS, atom("A", "B"), query)
    result = solver.analyze(problem)
    assert result.error == "z3 returned unknown: injected unknown"
    assert resolve_answer(result, AnswerMode.SINGLE).status is ResolutionStatus.ERROR
    assessment = solver.assess(problem, atom("A", "B"))
    assert assessment.error
    assert not assessment.entailed
    with pytest.raises(RuntimeError, match="injected unknown"):
        solver.semantic_signature(problem)


def test_restricted_direction_singleton_is_not_full_domain_entailment():
    pytest.importorskip("z3")
    claim = atom("A", "B")
    premise = Or((claim, atom("A", "B", Direction.NORTHWEST)))
    query = DirectionQuery("A", "B", claim.allowed)
    problem = SpatialProblem(OBJECTS, premise, query)
    solver = SpatialSolverV2(backend="z3")
    analysis = solver.analyze(problem)
    assert isinstance(analysis, DirectionAnalysis)
    assert analysis.possible_directions == (Direction.NORTHEAST,)
    assert not solver.assess(problem, claim).entailed
