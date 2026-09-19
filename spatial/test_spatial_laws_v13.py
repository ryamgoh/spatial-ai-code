"""Public-law tests for the v13 spatial solver.

Run with:
    uv run --python 3.12 --no-project --with pytest --with typer \
      pytest spatial/test_spatial_laws_v13.py -q
"""

from __future__ import annotations

from spatial_solver_v13 import SpatialSolverV13


SOLVER = SpatialSolverV13()


def prompt(sentences: list[str], question: str, options: dict[str, str]) -> str:
    rendered_options = ", ".join(f"{key}. {value}" for key, value in options.items())
    return (
        "Consider a map with multiple locations:\n\n"
        + " ".join(sentences)
        + f"\n\nQuestion: {question} Available options: {rendered_options}"
    )


DIR_OPTIONS = {
    "A": "Northeast",
    "B": "Northwest",
    "C": "Southeast",
    "D": "Southwest",
    "E": "Cannot be determined",
}


def test_cardinal_facts_compose_into_a_compound_direction() -> None:
    text = prompt(
        [
            "The Library is to the East of the Bank.",
            "The Library is to the North of the Bank.",
        ],
        "In which direction is the Library relative to the Bank?",
        DIR_OPTIONS,
    )

    assert SOLVER.solve(text) == "A"


def test_cardinal_relations_work_for_which_and_count_questions() -> None:
    sentences = [
        "The Library is to the East of the Bank.",
        "The Museum is to the East of the Library.",
        "The Park is to the West of the Bank.",
    ]
    which_text = prompt(
        sentences,
        "Which object is in the East of the Bank?",
        {
            "A": "Library",
            "B": "Museum",
            "C": "Park",
            "D": "Bank",
            "E": "None of the Options",
        },
    )
    count_text = prompt(
        sentences,
        "How many objects are in the East of the Bank?",
        {"A": "0", "B": "1", "C": "2", "D": "3", "E": "None of the Options"},
    )

    assert SOLVER.solve(which_text) == "A,B"
    assert SOLVER.solve(count_text) == "C"


def test_analysis_reports_independent_shortest_axis_proofs() -> None:
    text = prompt(
        [
            "The Museum is to the East of the Bank.",
            "The Library is to the East of the Museum.",
            "The Park is to the North of the Bank.",
            "The Library is to the North of the Park.",
        ],
        "In which direction is the Library relative to the Bank?",
        DIR_OPTIONS,
    )

    analysis = SOLVER.analyze(text)

    assert analysis["relation_mix"] == "cardinal-only"
    assert analysis["x_depth"] == 2
    assert analysis["y_depth"] == 2
    assert analysis["x_direct"] is False
    assert analysis["y_direct"] is False
    assert analysis["axes_independent"] is True
    assert analysis["x_path"] == ["Bank", "Museum", "Library"]
    assert analysis["y_path"] == ["Bank", "Park", "Library"]


def test_analysis_marks_shared_diagonal_evidence() -> None:
    text = prompt(
        ["The Library is to the Northeast of the Bank."],
        "In which direction is the Library relative to the Bank?",
        DIR_OPTIONS,
    )

    analysis = SOLVER.analyze(text)

    assert SOLVER.solve(text) == "A"
    assert analysis["relation_mix"] == "diagonal-only"
    assert analysis["x_depth"] == 1
    assert analysis["y_depth"] == 1
    assert analysis["axes_independent"] is False
    assert analysis["shared_supporting_statements"] == [0]


def test_analysis_keeps_unknown_axis_depth_empty() -> None:
    text = prompt(
        ["The Library is to the East of the Bank."],
        "In which direction is the Library relative to the Bank?",
        DIR_OPTIONS,
    )

    analysis = SOLVER.analyze(text)

    assert SOLVER.solve(text) == "A,C"
    assert analysis["x_depth"] == 1
    assert analysis["y_depth"] is None
    assert analysis["axes_independent"] is None


def test_cardinal_cycle_is_not_a_definite_order() -> None:
    text = prompt(
        [
            "The Library is to the East of the Bank.",
            "The Bank is to the East of the Library.",
        ],
        "In which direction is the Library relative to the Bank?",
        DIR_OPTIONS,
    )

    grade = SOLVER.grade(text)
    analysis = SOLVER.analyze(text)

    assert grade.raw == "E"
    assert grade.x_conflict is True
    assert analysis["x_conflict"] is True


def test_solve_and_analyze_returns_one_authoritative_result() -> None:
    text = prompt(
        [
            "The Museum is to the East of the Bank.",
            "The Library is to the East of the Museum.",
            "The Library is to the North of the Bank.",
        ],
        "In which direction is the Library relative to the Bank?",
        DIR_OPTIONS,
    )

    solved = SOLVER.solve_and_analyze(text)

    assert solved.grade.raw == "A"
    assert solved.structure["x_depth"] == 2
    assert solved.structure["y_depth"] == 1
    assert solved.structure == SOLVER.analyze(text)


def test_reported_paths_follow_low_to_high_axis_order_for_west_and_south() -> None:
    text = prompt(
        [
            "The Museum is to the West of the Bank.",
            "The Library is to the West of the Museum.",
            "The Park is to the South of the Bank.",
            "The Library is to the South of the Park.",
        ],
        "In which direction is the Library relative to the Bank?",
        DIR_OPTIONS,
    )

    solved = SOLVER.solve_and_analyze(text)

    assert solved.grade.x_rel == "lt"
    assert solved.grade.y_rel == "lt"
    assert solved.structure["x_path"] == ["Library", "Museum", "Bank"]
    assert solved.structure["y_path"] == ["Library", "Park", "Bank"]
