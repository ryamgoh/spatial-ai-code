"""Contract tests for the exact eight-direction V2 solver."""

from __future__ import annotations

import random
from itertools import permutations, product

import pytest

from spatial.v2.grading import (
    AnswerMode,
    MenuAnswer,
    encode_menu_answer,
    resolve_answer,
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
)
from spatial.v2.spatialeval_adapter import SpatialEvalAdapter, audit
from spatial.v2.text import SpatialTextAdapter

SOLVER = SpatialSolverV2()
TEXT_ADAPTER = SpatialTextAdapter()
SPATIALEVAL_ADAPTER = SpatialEvalAdapter(TEXT_ADAPTER)
ALL_OPTIONS = {
    "A": "North",
    "B": "Northeast",
    "C": "East",
    "D": "Southeast",
    "E": "South",
    "F": "Southwest",
    "G": "West",
    "H": "Northwest",
    "I": "Cannot be determined",
    "J": "None of the Options",
}
COARSE = {
    "Northward": {Direction.NORTHWEST, Direction.NORTH, Direction.NORTHEAST},
    "Eastward": {Direction.NORTHEAST, Direction.EAST, Direction.SOUTHEAST},
    "Southward": {Direction.SOUTHEAST, Direction.SOUTH, Direction.SOUTHWEST},
    "Westward": {Direction.SOUTHWEST, Direction.WEST, Direction.NORTHWEST},
}
SIGNS_DIRECTION = {
    (0, 1): Direction.NORTH,
    (1, 1): Direction.NORTHEAST,
    (1, 0): Direction.EAST,
    (1, -1): Direction.SOUTHEAST,
    (0, -1): Direction.SOUTH,
    (-1, -1): Direction.SOUTHWEST,
    (-1, 0): Direction.WEST,
    (-1, 1): Direction.NORTHWEST,
}


def prompt(sentences: list[str], target: str = "A", reference: str = "B") -> str:
    rendered_options = ", ".join(
        f"{key}. {value}" for key, value in ALL_OPTIONS.items()
    )
    return (
        "Consider a map with multiple locations:\n\n"
        + " ".join(sentences)
        + f"\n\nQuestion: In which direction is {target} relative to {reference}? "
        + f"Available options: {rendered_options}"
    )


def spatialeval_prompt(
    sentences: list[str], target: str = "A", reference: str = "B"
) -> str:
    return (
        "Consider a map with multiple objects:\n"
        + " ".join(sentences)
        + "\n\nPlease answer the following multiple-choice question based on the "
        + f"provided information. In which direction is {target} relative to "
        + f"{reference}? Available options:\nA. Northeast\nB. Northwest\n"
        + "C. Southwest\nD. Southeast."
    )


def which_prompt(sentences: list[str]) -> str:
    return (
        "Consider a map with multiple objects:\n"
        + " ".join(sentences)
        + "\n\nQuestion: Which object is in the Northeast of B? Available options:\n"
        + "A. A\nB. C\nC. D\nD. None of the Options\nE. Cannot be determined."
    )


def count_prompt(sentences: list[str]) -> str:
    return (
        "Consider a map with multiple objects:\n"
        + " ".join(sentences)
        + "\n\nQuestion: How many objects are in the Northeast of B? Available options:\n"
        + "A. 0\nB. 1\nC. 2\nD. None of the Options\nE. Cannot be determined."
    )


def possible(text: str) -> set[Direction]:
    return set(analyze_text(text).possible_directions)


def analyze_text(text: str, backend: str | None = None) -> DirectionAnalysis:
    parsed = TEXT_ADAPTER.parse(text)
    solver = SOLVER if backend is None else SpatialSolverV2(backend=backend)
    return solver.analyze(parsed.problem)


def menu_answer_text(
    text: str,
    answer_mode: AnswerMode = AnswerMode.ALL_POSSIBLE,
) -> MenuAnswer:
    parsed = TEXT_ADAPTER.parse(text)
    resolution = resolve_answer(SOLVER.analyze(parsed.problem), answer_mode)
    return encode_menu_answer(resolution, parsed.options)


def solve_text(
    text: str,
    answer_mode: AnswerMode = AnswerMode.ALL_POSSIBLE,
) -> str:
    return menu_answer_text(text, answer_mode).raw


def coordinate_direction(
    subject: tuple[int, int], reference: tuple[int, int]
) -> Direction:
    difference = (
        (subject[0] > reference[0]) - (subject[0] < reference[0]),
        (subject[1] > reference[1]) - (subject[1] < reference[1]),
    )
    return SIGNS_DIRECTION[difference]


def atom(subject: str, direction: Direction, reference: str) -> RelationConstraint:
    return RelationConstraint(subject, reference, frozenset({direction}))


def brute_force_directions(
    constraints: list[tuple[str, str, bool, str]],
) -> set[Direction]:
    names = ("A", "B", "C")
    points = tuple(product(range(len(names)), repeat=2))
    possible_directions: set[Direction] = set()
    for assigned_points in permutations(points, len(names)):
        coordinates = dict(zip(names, assigned_points, strict=True))
        for subject, relation, negative, reference in constraints:
            actual = coordinate_direction(coordinates[subject], coordinates[reference])
            allowed = COARSE[relation] if relation in COARSE else {Direction(relation)}
            if (actual in allowed) == negative:
                break
        else:
            possible_directions.add(
                coordinate_direction(coordinates["A"], coordinates["B"])
            )
    return possible_directions


def test_reasoning_core_accepts_structured_problems_only() -> None:
    problem = SpatialProblem(
        objects=("A", "B", "C"),
        premise=And(
            (
                RelationConstraint("A", "B", frozenset({Direction.NORTH})),
                RelationConstraint("B", "C", frozenset({Direction.EAST})),
            )
        ),
        query=DirectionQuery("A", "C"),
    )

    analysis = SOLVER.analyze(problem)

    assert analysis.possible_directions == (Direction.NORTHEAST,)
    with pytest.raises(TypeError, match="SpatialProblem"):
        SOLVER.analyze("not a structured problem")  # type: ignore[arg-type]


def test_problem_candidate_directions_restrict_the_generic_solver() -> None:
    diagonals = frozenset(
        {
            Direction.NORTHEAST,
            Direction.NORTHWEST,
            Direction.SOUTHEAST,
            Direction.SOUTHWEST,
        }
    )
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=RelationConstraint("A", "B", diagonals),
        query=DirectionQuery("A", "B", diagonals),
    )

    analysis = SOLVER.analyze(problem)

    assert set(analysis.possible_directions) == diagonals
    assert all(
        coordinates["A"][0] != coordinates["B"][0]
        and coordinates["A"][1] != coordinates["B"][1]
        for coordinates in analysis.witnesses.values()
    )


@pytest.mark.parametrize("backend", ["reference", "z3"])
def test_propositional_disjunctive_syllogism(backend: str) -> None:
    if backend == "z3":
        pytest.importorskip("z3")
    northeast = atom("A", Direction.NORTHEAST, "B")
    northwest = atom("A", Direction.NORTHWEST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=And((Or((northeast, northwest)), Not(northeast))),
        query=DirectionQuery("A", "B"),
    )

    analysis = SpatialSolverV2(backend=backend).analyze(problem)

    assert analysis.possible_directions == (Direction.NORTHWEST,)


@pytest.mark.parametrize(
    "formula_builder",
    [
        lambda premise, conclusion: And((Implies(premise, conclusion), premise)),
        lambda premise, conclusion: And((Iff(premise, conclusion), premise)),
    ],
)
@pytest.mark.parametrize("backend", ["reference", "z3"])
def test_implication_and_biconditional_reasoning(backend: str, formula_builder) -> None:
    if backend == "z3":
        pytest.importorskip("z3")
    premise = atom("A", Direction.NORTHEAST, "B")
    conclusion = atom("C", Direction.SOUTHWEST, "D")
    problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=formula_builder(premise, conclusion),
        query=DirectionQuery("C", "D"),
    )

    analysis = SpatialSolverV2(backend=backend).analyze(problem)

    assert analysis.possible_directions == (Direction.SOUTHWEST,)


def test_empty_boolean_connective_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires at least one operand"):
        SpatialProblem(
            objects=("A", "B"),
            premise=And(()),
            query=DirectionQuery("A", "B"),
        )


def test_spatialeval_adapter_reports_constructive_ambiguity() -> None:
    row = {
        "id": "spatialmap.tqa.audit",
        "text": spatialeval_prompt(
            [
                "A is to the Northeast of B.",
                "B is to the Northwest of C.",
            ],
            target="A",
            reference="C",
        ),
        "oracle_answer": "Northeast",
        "oracle_option": "A",
    }

    case = SPATIALEVAL_ADAPTER.parse(row)
    result = audit(case, SOLVER.analyze(case.problem))

    assert result.status == "underdetermined-oracle-possible"
    assert result.case.answer_mode is AnswerMode.SINGLE
    assert result.menu_answer.status == "ambiguous"
    assert result.menu_answer.raw == "Error: menu has no undetermined option"
    assert set(result.analysis.witnesses) == {
        Direction.NORTHWEST,
        Direction.NORTHEAST,
    }


def test_spatialeval_adapter_preserves_an_exact_oracle_match() -> None:
    row = {
        "id": "spatialmap.tqa.valid",
        "text": spatialeval_prompt(["A is to the Northeast of B."]),
        "oracle_answer": "Northeast",
        "oracle_option": "A",
    }

    case = SPATIALEVAL_ADAPTER.parse(row)
    result = audit(case, SOLVER.analyze(case.problem))

    assert result.status == "exact-match"
    assert result.analysis.possible_directions == (Direction.NORTHEAST,)


def test_spatialeval_possible_but_unentailed_which_oracle_is_not_exact() -> None:
    row = {
        "id": "spatialmap.tqa.which-contingent",
        "text": which_prompt(
            [
                "A is to the Northeast of X.",
                "X is to the Northwest of B.",
                "C is to the Southwest of B.",
                "D is to the Southeast of B.",
            ]
        ),
        "oracle_answer": "A",
        "oracle_option": "A",
    }

    case = SPATIALEVAL_ADAPTER.parse(row)
    result = audit(case, SOLVER.analyze(case.problem))

    assert result.analysis.possible_entities == ("A",)
    assert result.analysis.entailed_entities == ()
    assert result.status == "underdetermined-oracle-possible"
    assert result.menu_answer.status == "undetermined"
    assert result.menu_answer.raw == "E"


def test_spatialeval_adapter_rejects_non_ordinal_premises() -> None:
    row = {
        "id": "spatialmap.tqa.invalid-dialect",
        "text": spatialeval_prompt(["A is to the Northward of B."]),
        "oracle_answer": "Northeast",
        "oracle_option": "A",
    }

    with pytest.raises(ValueError, match="exact ordinal"):
        SPATIALEVAL_ADAPTER.parse(row)


@pytest.mark.parametrize(
    "text,query_type",
    [
        (which_prompt(["A is to the Northeast of B."]), WhichQuery),
        (
            count_prompt(["A is to the Northeast of B."]).replace(
                "Northeast of B?", "North of B?"
            ),
            CountQuery,
        ),
    ],
)
def test_spatialeval_adapter_keeps_all_compass_query_labels(
    text: str, query_type
) -> None:
    row = {
        "id": "spatialmap.tqa.query-family",
        "text": text,
        "oracle_answer": "A" if query_type is WhichQuery else "0",
        "oracle_option": "A",
    }

    case = SPATIALEVAL_ADAPTER.parse(row)

    assert isinstance(case.problem.query, query_type)
    if query_type is CountQuery:
        assert case.problem.query.directions == frozenset({Direction.NORTH})


@pytest.mark.parametrize("backend", ["reference", "z3"])
def test_which_query_distinguishes_possible_and_entailed_entities(
    backend: str,
) -> None:
    parsed = TEXT_ADAPTER.parse(
        which_prompt(
            [
                "A is to the Northeast of B.",
                "C is to the Northwest of A.",
                "D is to the Southwest of B.",
            ]
        )
    )

    analysis = SpatialSolverV2(backend=backend).analyze(parsed.problem)

    assert isinstance(analysis, WhichAnalysis)
    assert analysis.possible_entities == ("A", "C")
    assert analysis.entailed_entities == ("A",)
    assert analysis.contingent_entities == ("C",)
    assert analysis.impossible_entities == ("D",)
    exact = encode_menu_answer(
        resolve_answer(analysis, AnswerMode.SINGLE),
        parsed.options,
    )
    possible = encode_menu_answer(
        resolve_answer(analysis, AnswerMode.ALL_POSSIBLE),
        parsed.options,
    )
    assert exact.raw == "E"
    assert possible.raw == "A,B"


@pytest.mark.parametrize("backend", ["reference", "z3"])
def test_count_query_returns_possible_correlated_counts(backend: str) -> None:
    parsed = TEXT_ADAPTER.parse(
        count_prompt(
            [
                "A is to the Northeast of B.",
                "C is to the Northwest of A.",
            ]
        )
    )

    analysis = SpatialSolverV2(backend=backend).analyze(parsed.problem)

    assert isinstance(analysis, CountAnalysis)
    assert analysis.possible_counts == (1, 2)
    exact = encode_menu_answer(
        resolve_answer(analysis, AnswerMode.SINGLE),
        parsed.options,
    )
    possible = encode_menu_answer(
        resolve_answer(analysis, AnswerMode.ALL_POSSIBLE),
        parsed.options,
    )
    assert exact.raw == "E"
    assert possible.raw == "B,C"


@pytest.mark.parametrize("backend", ["reference", "z3"])
def test_count_query_preserves_membership_correlations(backend: str) -> None:
    a_is_ne = atom("A", Direction.NORTHEAST, "B")
    c_is_ne = atom("C", Direction.NORTHEAST, "B")
    problem = SpatialProblem(
        objects=("A", "B", "C"),
        premise=And((Or((a_is_ne, c_is_ne)), Not(And((a_is_ne, c_is_ne))))),
        query=CountQuery(frozenset({Direction.NORTHEAST}), "B", ("A", "C")),
    )

    analysis = SpatialSolverV2(backend=backend).analyze(problem)

    assert isinstance(analysis, CountAnalysis)
    assert analysis.possible_counts == (1,)
    assert set(analysis.witnesses) == {1}


def test_exact_cardinals_retain_axis_equality() -> None:
    text = prompt(["A is to the North of B."])

    assert possible(text) == {Direction.NORTH}
    assert solve_text(text) == "A"


def test_exact_cardinals_compose_into_a_diagonal() -> None:
    text = prompt(
        [
            "A is to the North of B.",
            "B is to the East of C.",
        ],
        target="A",
        reference="C",
    )

    assert possible(text) == {Direction.NORTHEAST}
    assert solve_text(text) == "B"


def test_negated_diagonal_preserves_all_other_seven_directions() -> None:
    text = prompt(["A is not Northeast of B."])

    assert possible(text) == set(Direction) - {Direction.NORTHEAST}
    assert solve_text(text) == "A,C,D,E,F,G,H"


def test_coarse_eastward_plus_not_northeast_leaves_east_or_southeast() -> None:
    text = prompt(
        [
            "A is to the Eastward of B.",
            "A is not to the Northeast of B.",
        ]
    )

    assert possible(text) == {Direction.EAST, Direction.SOUTHEAST}
    assert solve_text(text) == "C,D"


def test_answer_mode_distinguishes_multi_complete_from_single() -> None:
    text = prompt(["A is to the Eastward of B."])

    possible_set = menu_answer_text(
        text,
        AnswerMode.ALL_POSSIBLE,
    )
    single_exact = menu_answer_text(
        text,
        AnswerMode.SINGLE,
    )

    assert possible_set.raw == "B,C,D"
    assert possible_set.status == "possibilities"
    assert possible_set.resolution.mode is AnswerMode.ALL_POSSIBLE
    assert single_exact.raw == "I"
    assert single_exact.status == "undetermined"
    assert single_exact.resolution.mode is AnswerMode.SINGLE


@pytest.mark.parametrize(
    "answer_mode",
    [AnswerMode.ALL_POSSIBLE, AnswerMode.SINGLE],
)
def test_unique_answer_is_the_same_under_single_and_multi_complete(
    answer_mode: AnswerMode,
) -> None:
    text = prompt(["A is to the East of B."])

    result = menu_answer_text(text, answer_mode)

    assert result.raw == "C"
    assert result.status == (
        "exact" if answer_mode is AnswerMode.SINGLE else "possibilities"
    )


def test_atomic_negation_can_expose_a_transitive_contradiction() -> None:
    text = prompt(
        [
            "A is to the East of B.",
            "B is to the East of C.",
            "A is not to the East of C.",
        ]
    )

    analysis = analyze_text(text)
    result = menu_answer_text(text)
    assert analysis.consistent is False
    assert result.raw == "Error: premises are inconsistent"
    assert result.status == "inconsistent"


def test_two_distinct_objects_cannot_be_equal_on_both_axes() -> None:
    text = prompt(
        [
            "A is to the North of B.",
            "A is to the East of B.",
        ]
    )

    assert analyze_text(text).consistent is False


def test_inverse_query_is_derived_from_an_exact_relation() -> None:
    text = prompt(["A is to the Southeast of B."], target="B", reference="A")

    assert possible(text) == {Direction.NORTHWEST}
    assert solve_text(text) == "H"


def test_missing_information_is_not_treated_as_negation() -> None:
    text = prompt(
        [
            "A is to the Northward of C.",
            "B is to the Southward of C.",
        ],
        target="A",
        reference="B",
    )

    # Y is known, but X may be less, equal, or greater.
    analysis = analyze_text(text)
    expected = {
        Direction.NORTHWEST,
        Direction.NORTH,
        Direction.NORTHEAST,
    }
    assert set(analysis.possible_directions) == expected
    assert set(analysis.witnesses) == expected
    for direction, coordinates in analysis.witnesses.items():
        assert coordinate_direction(coordinates["A"], coordinates["B"]) is direction
    assert solve_text(text) == "A,B,H"


def test_partial_multi_complete_menu_uses_undetermined_without_dropping_values() -> (
    None
):
    text = prompt(["A is to the Northward of B."])
    text = text.replace(
        ", B. Northeast, C. East, D. Southeast, E. South, F. Southwest, "
        "G. West, H. Northwest",
        "",
    )

    result = menu_answer_text(text)
    assert result.raw == "I"
    assert result.status == "incomplete-menu"


def test_malformed_premise_fails_closed() -> None:
    text = prompt(["A floats mysteriously beside B."])

    with pytest.raises(ValueError, match="cannot parse premise"):
        TEXT_ADAPTER.parse(text)


def test_question_object_must_exist_in_the_premises() -> None:
    text = prompt(["A is to the North of B."], target="Ghost", reference="B")

    with pytest.raises(ValueError, match="query references an object absent"):
        TEXT_ADAPTER.parse(text)


def test_options_may_be_rendered_on_separate_lines() -> None:
    text = prompt(["A is to the East of B."])
    for key in tuple(ALL_OPTIONS)[1:]:
        text = text.replace(f", {key}.", f"\n{key}.")

    assert solve_text(text) == "C"


@pytest.mark.parametrize(
    "sentences,target,reference",
    [
        (["A is not Northeast of B."], "A", "B"),
        (
            [
                "A is to the Eastward of B.",
                "A is not to the Northeast of B.",
            ],
            "A",
            "B",
        ),
        (
            [
                "A is to the North of B.",
                "B is to the East of C.",
            ],
            "A",
            "C",
        ),
        (
            [
                "A is to the East of B.",
                "B is to the East of C.",
                "A is not to the East of C.",
            ],
            "A",
            "B",
        ),
    ],
)
def test_z3_and_reference_engines_agree(
    sentences: list[str], target: str, reference: str
) -> None:
    pytest.importorskip("z3")
    text = prompt(sentences, target, reference)

    expected = analyze_text(text, backend="reference")
    actual = analyze_text(text, backend="z3")

    assert actual.consistent == expected.consistent
    assert actual.possible_directions == expected.possible_directions


def test_z3_handles_a_long_disjunctive_chain() -> None:
    pytest.importorskip("z3")
    sentences = [f"P{index} is to the Eastward of P{index + 1}." for index in range(31)]
    sentences.extend(
        f"P{index} is not Northeast of P{index + 1}." for index in range(31)
    )
    text = prompt(sentences, target="P0", reference="P31")

    analysis = analyze_text(text, backend="z3")

    assert analysis.consistent is True
    assert set(analysis.possible_directions) == {
        Direction.EAST,
        Direction.SOUTHEAST,
    }


def test_z3_matches_reference_on_seeded_mixed_constraints() -> None:
    pytest.importorskip("z3")
    rng = random.Random(20260927)
    objects = ("A", "B", "C", "D")
    relations = tuple(direction.value for direction in Direction) + (
        "Northward",
        "Eastward",
        "Southward",
        "Westward",
    )
    reference = SpatialSolverV2(backend="reference")
    z3_solver = SpatialSolverV2(backend="z3")

    for _ in range(50):
        sentences = ["A is in the map.", "B is in the map."]
        for _ in range(3):
            subject, relation_reference = rng.sample(objects, 2)
            negative = "not " if rng.random() < 0.5 else "to the "
            relation = rng.choice(relations)
            sentences.append(
                f"{subject} is {negative}{relation} of {relation_reference}."
            )
        text = prompt(sentences)

        problem = TEXT_ADAPTER.parse(text).problem
        expected = reference.analyze(problem)
        actual = z3_solver.analyze(problem)

        assert actual.consistent == expected.consistent
        assert actual.possible_directions == expected.possible_directions


def test_z3_matches_exhaustive_coordinate_models() -> None:
    pytest.importorskip("z3")
    rng = random.Random(20260928)
    objects = ("A", "B", "C")
    relations = tuple(direction.value for direction in Direction) + tuple(COARSE)
    solver = SpatialSolverV2(backend="z3")

    for _ in range(50):
        constraints = []
        sentences = ["A is in the map.", "B is in the map."]
        for _ in range(4):
            subject, reference = rng.sample(objects, 2)
            relation = rng.choice(relations)
            negative = rng.random() < 0.5
            constraints.append((subject, relation, negative, reference))
            wording = "not " if negative else "to the "
            sentences.append(f"{subject} is {wording}{relation} of {reference}.")
        expected = brute_force_directions(constraints)

        analysis = solver.analyze(TEXT_ADAPTER.parse(prompt(sentences)).problem)

        assert analysis.consistent is bool(expected)
        assert set(analysis.possible_directions) == expected
        if not expected:
            assert analysis.coordinates is None
            assert analysis.witnesses == {}
            continue
        assert analysis.coordinates is not None
        assert set(analysis.witnesses) == expected
        for query_direction, coordinates in analysis.witnesses.items():
            assert (
                coordinate_direction(coordinates["A"], coordinates["B"])
                is query_direction
            )
            for subject, relation, negative, reference in constraints:
                actual = coordinate_direction(
                    coordinates[subject], coordinates[reference]
                )
                allowed = (
                    COARSE[relation] if relation in COARSE else {Direction(relation)}
                )
                assert (actual in allowed) != negative


@pytest.mark.parametrize("backend", ["reference", "z3"])
def test_analysis_returns_a_compact_coordinate_witness(backend: str) -> None:
    if backend == "z3":
        pytest.importorskip("z3")
    text = prompt(
        [
            "A is to the North of B.",
            "B is to the East of C.",
        ],
        target="A",
        reference="C",
    )

    analysis = analyze_text(text, backend=backend)

    assert analysis.coordinates is not None
    assert (
        coordinate_direction(analysis.coordinates["A"], analysis.coordinates["C"])
        is Direction.NORTHEAST
    )
    for axis in (0, 1):
        values = sorted({point[axis] for point in analysis.coordinates.values()})
        assert values == list(range(len(values)))


def test_inconsistent_world_has_no_coordinate_witness() -> None:
    text = prompt(
        [
            "A is to the East of B.",
            "A is not East of B.",
        ]
    )

    analysis = analyze_text(text, backend="z3")

    assert analysis.consistent is False
    assert analysis.coordinates is None
    assert analysis.witnesses == {}
