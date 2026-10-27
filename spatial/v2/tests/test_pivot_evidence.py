"""Executable semantic fixtures for spatialentail-composition-v1.

See experiments/spatialentail-composition-v1/research-analysis/PIVOT-EVIDENCE.md.

The menu mapping below is a specification oracle, not the historical grader.
It must not be mistaken for a migrated production encoder or prompt parser.
"""

from dataclasses import dataclass, replace
from itertools import permutations

import pytest

from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    RelationConstraint,
    SpatialProblem,
    SpatialQuery,
    SpatialSolverV2,
    WhichAnalysis,
    WhichQuery,
)
from spatial.v2.text import SpatialTextAdapter, render_spatial_formula


@dataclass(frozen=True)
class Case:
    name: str
    objects: tuple[str, ...]
    facts: tuple[tuple[str, str, str], ...]
    query: SpatialQuery
    possible: frozenset
    certain: frozenset = frozenset()
    consistent: bool = True

    def problem(self):
        return SpatialProblem(
            self.objects,
            And(
                tuple(
                    RelationConstraint(a, b, frozenset({Direction(d)}))
                    for a, d, b in self.facts
                )
            ),
            self.query,
        )


NORTH = frozenset({Direction.NORTH})
ELIGIBLE = ("U", "V", "W", "T")
FIVE_ELIGIBLE = (*ELIGIBLE, "Y")
SELECTION = WhichQuery(NORTH, "O", ELIGIBLE)
COUNT = CountQuery(NORTH, "O", ELIGIBLE)
NO_MATCH = (
    ("U", "East", "O"),
    ("V", "West", "O"),
    ("W", "South", "O"),
    ("T", "Southeast", "O"),
)
MULTIPLE = (
    ("U", "North", "O"),
    ("V", "North", "U"),
    ("W", "East", "O"),
    ("T", "South", "O"),
)
CASES = (
    Case(
        "D1",
        ("X", "Y", "Z"),
        (("X", "Northeast", "Y"), ("Z", "South", "Y")),
        DirectionQuery("X", "Z"),
        frozenset({"Northeast"}),
    ),
    Case(
        "D2",
        ("X", "Y", "O"),
        (("X", "East", "O"), ("Y", "East", "O")),
        DirectionQuery("X", "Y"),
        frozenset({"East", "West"}),
    ),
    Case(
        "S1",
        ("O", "Q", *ELIGIBLE),
        (
            ("U", "North", "Q"),
            ("Q", "North", "O"),
            ("V", "East", "O"),
            ("W", "South", "O"),
            ("T", "West", "O"),
        ),
        SELECTION,
        frozenset({"U"}),
        frozenset({"U"}),
    ),
    Case("S2", ("O", *ELIGIBLE), NO_MATCH, SELECTION, frozenset()),
    Case(
        "S3",
        ("O", *ELIGIBLE),
        MULTIPLE,
        SELECTION,
        frozenset({"U", "V"}),
        frozenset({"U", "V"}),
    ),
    Case(
        "S4",
        ("O", *ELIGIBLE),
        (
            ("U", "Northeast", "V"),
            ("O", "Northeast", "V"),
            ("W", "East", "O"),
            ("T", "South", "O"),
        ),
        SELECTION,
        frozenset({"U"}),
    ),
    Case(
        "S5",
        ("O", *FIVE_ELIGIBLE),
        tuple((obj, "North", "O") for obj in FIVE_ELIGIBLE),
        WhichQuery(NORTH, "O", FIVE_ELIGIBLE),
        frozenset(FIVE_ELIGIBLE),
        frozenset(FIVE_ELIGIBLE),
    ),
    Case(
        "S6",
        ("O", *FIVE_ELIGIBLE),
        (*NO_MATCH, ("Y", "North", "O")),
        WhichQuery(NORTH, "O", FIVE_ELIGIBLE),
        frozenset({"Y"}),
        frozenset({"Y"}),
    ),
    Case(
        "C1",
        ("O", *ELIGIBLE),
        (("U", "East", "V"), ("W", "South", "O"), ("T", "South", "W")),
        COUNT,
        frozenset({0, 1}),
    ),
    Case("C2", ("O", *ELIGIBLE), MULTIPLE, COUNT, frozenset({2})),
    Case("C3", ("O", *ELIGIBLE), NO_MATCH, COUNT, frozenset({0})),
    Case(
        "I1",
        ("X", "Y", "Z", "U", "V"),
        (
            ("X", "East", "Y"),
            ("Y", "East", "Z"),
            ("Z", "East", "X"),
            ("U", "North", "V"),
        ),
        DirectionQuery("U", "V"),
        frozenset(),
        consistent=False,
    ),
    Case(
        "I2",
        ("X", "Y", "Z", "U", "V"),
        (
            ("X", "East", "Y"),
            ("Y", "East", "Z"),
            ("X", "East", "Z"),
            ("U", "North", "V"),
        ),
        DirectionQuery("U", "V"),
        frozenset({"North"}),
    ),
    Case(
        "I3",
        ("X", "Y", "O", "T"),
        (
            ("X", "North", "O"),
            ("Y", "North", "O"),
            ("X", "East", "T"),
            ("Y", "East", "T"),
        ),
        DirectionQuery("X", "O"),
        frozenset(),
        consistent=False,
    ),
)
BY_ID = {case.name: case for case in CASES}

# Each menu has four substantive values; E/F follow the fixed pivot wording.
VIEWS = (
    ("D1", ("West", "Northeast", "South", "Northwest"), ("B", "B", "B")),
    ("D1", ("North", "East", "South", "West"), ("F", "F", "F")),
    ("D2", ("North", "East", "South", "Northeast"), ("E", "B", "F")),
    ("D2", ("West", "East", "South", "Northeast"), ("E", "AB", "AB")),
    ("S1", ELIGIBLE, ("A", "A", "A")),
    ("S2", ELIGIBLE, ("E", "F", "F")),
    ("S3", ELIGIBLE, ("E", "AB", "AB")),
    ("S4", ELIGIBLE, ("E", "A", "A")),
    ("S5", ELIGIBLE, ("E", "ABCD", "F")),
    ("S6", ELIGIBLE, ("F", "F", "F")),
    ("S6", ("U", "V", "W", "Y"), ("D", "D", "D")),
    ("C1", (0, 1, 2, 3), ("E", "AB", "AB")),
    ("C1", (0, 2, 3, 4), ("E", "A", "F")),
    ("C2", (0, 1, 2, 3), ("C", "C", "C")),
    ("C2", (0, 1, 3, 4), ("F", "F", "F")),
    ("C3", (0, 1, 2, 3), ("A", "A", "A")),
    ("I1", ("North", "East", "South", "West"), ("E", "E", "E")),
    ("I2", ("North", "East", "South", "West"), ("A", "A", "A")),
    ("I3", ("North", "East", "South", "West"), ("E", "E", "E")),
)


def values(analysis):
    if hasattr(analysis, "possible_directions"):
        return frozenset(d.value for d in analysis.possible_directions)
    if hasattr(analysis, "possible_entities"):
        return frozenset(analysis.possible_entities)
    return frozenset(analysis.possible_counts)


def encode(case, menu, mode):
    if not case.consistent:
        return "E"
    possible = case.possible
    selected = "".join(
        letter for letter, value in zip("ABCD", menu) if value in possible
    )
    if mode == "unique":
        unique = len(possible) == 1
        if isinstance(case.query, WhichQuery):
            unique = unique and case.certain == possible
        return (selected or "F") if unique else "E"
    if mode == "listed":
        return selected or "F"
    return selected if possible and possible <= set(menu) else "F"


def check_map(case, coordinates):
    """Check coordinates independently of the solver's direction helpers."""
    assert set(coordinates) == set(case.objects)
    assert len(set(coordinates.values())) == len(case.objects)
    signs = {
        "North": (0, 1),
        "Northeast": (1, 1),
        "East": (1, 0),
        "Southeast": (1, -1),
        "South": (0, -1),
        "Southwest": (-1, -1),
        "West": (-1, 0),
        "Northwest": (-1, 1),
    }
    for subject, direction, reference in case.facts:
        observed = tuple(
            (a > b) - (a < b)
            for a, b in zip(coordinates[subject], coordinates[reference])
        )
        assert observed == signs[direction]


def north_members(coordinates, candidates=ELIGIBLE, reference="O"):
    ox, oy = coordinates[reference]
    return {
        obj
        for obj in candidates
        if coordinates[obj][0] == ox and coordinates[obj][1] > oy
    }


def check_witness_coverage(case, witnesses):
    answers = {
        answer.value if isinstance(answer, Direction) else answer
        for answer in witnesses
    }
    assert answers == case.possible, "Witnesses must cover every possible value"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_semantics_and_witnesses(case):
    solver = SpatialSolverV2(backend="z3")
    result = solver.analyze(case.problem(), include_witnesses=True)
    assert result.error is None
    assert result.consistent == case.consistent
    assert values(result) == case.possible
    if isinstance(case.query, WhichQuery):
        assert isinstance(result, WhichAnalysis)
        assert frozenset(result.entailed_entities) == case.certain
    check_witness_coverage(case, result.witnesses)
    for answer, coordinates in result.witnesses.items():
        check_map(case, coordinates)
        if isinstance(case.query, CountQuery):
            assert (
                len(
                    north_members(
                        coordinates, case.query.candidates, case.query.reference
                    )
                )
                == answer
            )
        elif isinstance(case.query, WhichQuery):
            assert answer in north_members(
                coordinates, case.query.candidates, case.query.reference
            )
        else:
            assert isinstance(answer, Direction)
            check_map(
                replace(
                    case,
                    facts=((case.query.target, answer.value, case.query.reference),),
                ),
                coordinates,
            )


@pytest.mark.parametrize("name", ("D2", "S5", "C1"))
@pytest.mark.parametrize("remove_all", (False, True))
def test_missing_witnesses_are_rejected(name, remove_all):
    case = BY_ID[name]
    result = SpatialSolverV2(backend="z3").analyze(
        case.problem(), include_witnesses=True
    )
    witnesses = dict(result.witnesses)
    check_witness_coverage(case, witnesses)
    if remove_all:
        witnesses.clear()
    else:
        witnesses.pop(next(iter(witnesses)))
    with pytest.raises(AssertionError, match="Witnesses must cover"):
        check_witness_coverage(case, witnesses)


@pytest.mark.parametrize("name,menu,expected", VIEWS)
def test_menu_truth_table_and_permutations(name, menu, expected):
    case = BY_ID[name]
    assert len(menu) == len(set(menu)) == 4
    if isinstance(case.query, WhichQuery):
        assert set(menu) <= set(case.query.candidates)
    elif isinstance(case.query, CountQuery):
        assert set(menu) <= set(range(len(case.query.candidates) + 1))
    else:
        assert set(menu) <= {direction.value for direction in Direction}
    modes = ("unique", "listed", "exhaustive")
    assert tuple(encode(case, menu, mode) for mode in modes) == expected
    for permuted in permutations(menu):
        for mode, original in zip(modes, expected):
            answer = encode(case, permuted, mode)
            if original in {"E", "F"}:
                assert answer == original
            else:
                original_values = {menu["ABCD".index(letter)] for letter in original}
                new_values = {permuted["ABCD".index(letter)] for letter in answer}
                assert new_values == original_values


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_premise_order_and_object_renaming(case):
    renamed = {obj: f"Object{i}" for i, obj in enumerate(case.objects)}
    query = case.query
    if isinstance(query, DirectionQuery):
        query = replace(
            query, target=renamed[query.target], reference=renamed[query.reference]
        )
    else:
        query = replace(
            query,
            reference=renamed[query.reference],
            candidates=tuple(renamed[obj] for obj in query.candidates),
        )
    variant = replace(
        case,
        objects=tuple(renamed[obj] for obj in case.objects),
        facts=tuple((renamed[a], d, renamed[b]) for a, d, b in reversed(case.facts)),
        query=query,
    )
    result = SpatialSolverV2(backend="z3").analyze(variant.problem())
    assert result.error is None
    assert result.consistent == case.consistent
    expected = (
        frozenset(renamed[obj] for obj in case.possible)
        if isinstance(query, WhichQuery)
        else case.possible
    )
    assert values(result) == expected


@pytest.mark.parametrize("name", ("D1", "D2"))
def test_small_reference_backend_agreement(name):
    case = BY_ID[name]
    result = SpatialSolverV2(backend="reference").analyze(case.problem())
    assert result.error is None
    assert values(result) == case.possible


def test_documented_counterexamples_and_joint_count():
    check_map(
        BY_ID["I2"],
        {"Z": (0, 0), "Y": (1, 0), "X": (2, 0), "V": (0, 2), "U": (0, 3)},
    )
    d2 = BY_ID["D2"]
    for x, y in ((1, 2), (2, 1)):
        check_map(d2, {"O": (0, 0), "X": (x, 0), "Y": (y, 0)})
    s4_base = {"O": (0, 0), "V": (-2, -2), "W": (2, 0), "T": (0, -2)}
    for u, expected in (((0, 1), {"U"}), ((1, 1), set())):
        coordinates = {**s4_base, "U": u}
        check_map(BY_ID["S4"], coordinates)
        assert north_members(coordinates) == expected
    c1_base = {"O": (0, 0), "W": (0, -1), "T": (0, -2)}
    for u, v, expected in (
        ((0, 1), (-1, 1), {"U"}),
        ((1, 1), (0, 1), {"V"}),
        ((2, 1), (1, 1), set()),
    ):
        coordinates = {**c1_base, "U": u, "V": v}
        check_map(BY_ID["C1"], coordinates)
        assert north_members(coordinates) == expected


@pytest.mark.parametrize(
    "name,coordinates,expected",
    (
        (
            "S5",
            {
                "O": (0, 0),
                "U": (0, 1),
                "V": (0, 2),
                "W": (0, 3),
                "T": (0, 4),
                "Y": (0, 5),
            },
            set(FIVE_ELIGIBLE),
        ),
        (
            "S6",
            {
                "O": (0, 0),
                "U": (1, 0),
                "V": (-1, 0),
                "W": (0, -1),
                "T": (1, -1),
                "Y": (0, 1),
            },
            {"Y"},
        ),
    ),
)
def test_documented_five_candidate_maps(name, coordinates, expected):
    case = BY_ID[name]
    check_map(case, coordinates)
    assert north_members(coordinates, case.query.candidates) == expected


@pytest.mark.parametrize("name,indices", (("I1", (0, 1, 2)), ("I3", (0, 1, 2, 3))))
def test_conflicting_subsets(name, indices):
    case = BY_ID[name]
    subset = replace(case, facts=tuple(case.facts[i] for i in indices))
    result = SpatialSolverV2(backend="z3").analyze(subset.problem())
    assert result.error is None
    assert not result.consistent


def test_d1_entailment_and_candidate_refutation():
    case = BY_ID["D1"]
    solver = SpatialSolverV2(backend="z3")
    for direction in Direction:
        result = solver.assess(
            case.problem(),
            RelationConstraint("X", "Z", frozenset({direction})),
        )
        assert result.error is None
        assert result.consistent
        assert result.possible == result.entailed == (direction is Direction.NORTHEAST)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_atomic_story_render_parse_round_trip(case):
    # Existing text adapter: validate the story using a neutral Direction query.
    # The new nine contract prompts and A-F decoder still need integration tests.
    target, reference = case.objects[:2]
    story = " ".join(
        render_spatial_formula(atom) + "." for atom in case.problem().premise.operands
    )
    text = (
        f"Consider a map with multiple locations:\n\n{story}\n\n"
        f"Question: In which direction is {target} relative to {reference}? "
        "Available options: A. North, B. East, C. South, D. West, "
        "E. Cannot be determined, F. None of the Options"
    )
    parsed = SpatialTextAdapter().parse(text).problem
    assert set(parsed.objects) == set(case.objects)
    assert parsed.premise == case.problem().premise
