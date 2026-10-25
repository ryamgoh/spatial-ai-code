"""Export reproducible renderer outputs for the professor brief.

Run from the repository root:
    .venv/bin/python fyp-report/docs/export_brief_traces.py
Add --check to compare against the saved asset without changing it.

These are checked fixture examples, not sampled or historical training rows.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spatial.v2.answer_certificates import (
    DirectionAnswerSetCertificate,
    check_direction_answer_set,
)
from spatial.v2.certificate_generation import build_answer_certificate
from spatial.v2.generation import GenerationPolicy, _render_trace
from spatial.v2.grading import AnswerMode, encode_menu_answer, resolve_answer
from spatial.v2.proof_renderers import render_direction_training_refutation
from spatial.v2.proofs import DirectionRefutationCertificate
from spatial.v2.solver import (
    And,
    Direction,
    DirectionAnalysis,
    DirectionQuery,
    SpatialProblem,
    SpatialSolverV2,
    direction_constraint,
)

OUTPUT = Path(__file__).with_name("brief-traces.json")


def export_example(problem, options, mode, fixture):
    assert isinstance(problem.query, DirectionQuery)
    solver = SpatialSolverV2(backend="reference")
    analysis = solver.analyze(problem)
    assert not analysis.error and analysis.consistent
    assert isinstance(analysis, DirectionAnalysis)
    certificate = build_answer_certificate(problem, analysis, solver)
    assert isinstance(certificate, DirectionAnswerSetCertificate)
    check_direction_answer_set(certificate)
    resolution = resolve_answer(analysis, mode)
    menu = encode_menu_answer(resolution, options)
    assert menu.is_resolved
    policy = GenerationPolicy(
        answer_mode=mode,
        menu_option_count=5,
        num_entities=len(problem.objects),
        num_premises=len(problem.premise.operands),
    )
    assert len(options) == policy.menu_option_count
    assert set(options.values()) <= {direction.value for direction in Direction}
    assert {direction.value for direction in analysis.possible_directions} <= set(
        options.values()
    )
    menu_analysis = solver.analyze(SpatialProblem(
        problem.objects,
        problem.premise,
        DirectionQuery(
            problem.query.target,
            problem.query.reference,
            frozenset(Direction(value) for value in options.values()),
        ),
    ))
    assert isinstance(menu_analysis, DirectionAnalysis)
    assert menu_analysis.possible_directions == analysis.possible_directions
    trace = _render_trace(certificate, resolution, menu, policy)
    # The completion wrapper matches GeneratedSpatialSample.as_sft_row.
    completion = (
        f"<think>\n{trace}\n</think>\nAnswer: "
        + ", ".join(sorted(menu.letters))
    )
    return {
        "fixture": fixture,
        "solver_backend": "reference",
        "solver_engine": analysis.engine,
        "options": options,
        "menu_text": " · ".join(
            f"{letter}. {value}" for letter, value in options.items()
        ),
        "answer_mode": mode.value,
        "completion": completion,
        "sha256": hashlib.sha256(completion.encode()).hexdigest(),
    }, certificate


def build_asset():
    northeast = Direction.NORTHEAST
    northwest = Direction.NORTHWEST
    southeast = Direction.SOUTHEAST
    southwest = Direction.SOUTHWEST
    g1, _ = export_example(
        SpatialProblem(
            ("A", "B"),
            And((direction_constraint("A", "B", northeast),)),
            DirectionQuery("A", "B"),
        ),
        {
            "A": "Northeast", "B": "Northwest", "C": "Southeast",
            "D": "Southwest", "E": "South",
        },
        AnswerMode.SINGLE,
        "test_unique_answer_set_covers_all_eight_candidates",
    )
    g2, certificate = export_example(
        SpatialProblem(
            ("N", "R", "S"),
            And((
                direction_constraint("S", "R", southeast),
                direction_constraint("N", "S", southwest),
            )),
            DirectionQuery("R", "N", frozenset({
                northeast, northwest, southeast, southwest,
            })),
        ),
        {
            "A": "Southeast", "B": "Southwest", "C": "Northeast",
            "D": "Northwest", "E": "South",
        },
        AnswerMode.ALL_POSSIBLE,
        "test_ambiguous_ordinal_answer_set_has_two_models_and_two_refutations",
    )
    candidate = next(
        item for item in certificate.candidates if item.direction is southeast
    )
    assert not candidate.possible
    assert isinstance(candidate.evidence, DirectionRefutationCertificate)
    refutation = render_direction_training_refutation(candidate.evidence)
    assert refutation in g2["completion"]
    # Split only at a candidate boundary; every completion line is retained.
    lines = g2["completion"].splitlines()
    split = lines.index("Southeast: impossible") + 1
    parts = ["\n".join(lines[:split]), "\n".join(lines[split:])]
    assert "\n".join(parts) == g2["completion"]
    g2["parts"] = parts
    g2["southeast_refutation"] = refutation
    return {
        "provenance": (
            "Checked fixture inputs, solved with the reference backend and "
            "rendered by production Natural renderers. Not historical training rows."
        ),
        "fixture_source": "spatial/v2/tests/test_answer_certificates.py",
        "g1": g1,
        "g2": g2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    asset = build_asset()
    if args.check:
        assert json.loads(OUTPUT.read_text()) == asset, "Saved traces differ from renderer"
        print("PASS: saved traces match regenerated output exactly.")
    else:
        OUTPUT.write_text(json.dumps(asset, indent=2) + "\n")
        print(f"Exported {OUTPUT}")
    for key in ("g1", "g2"):
        print(f"{key}: {asset[key]['sha256']}")
    print("PASS: certificates checked; G2 split is lossless; refutation is in G2.")


if __name__ == "__main__":
    main()
