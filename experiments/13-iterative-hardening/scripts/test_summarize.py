from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("summarize.py")
SPEC = importlib.util.spec_from_file_location("v13_summarize", MODULE_PATH)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def _sample(text: str, cell: str, gold: str, predicted: str, q_type: int) -> dict:
    inconsistent = cell.startswith("cycle-") and not cell.endswith("-control")
    return {
        "doc": {
            "text": text,
            "oracle_option": gold,
            "generation_cell": cell,
            "difficulty": {
                "question_type": q_type,
                "semantic_subtype": "dir-1",
                "relation_mix": "mixed",
                "world_consistency": (
                    "inconsistent" if inconsistent else "consistent"
                ),
            },
        },
        "filtered_resps": [predicted],
    }


def test_summary_reports_structural_buckets_and_paired_errors(tmp_path) -> None:
    cells = [
        _sample("q1", "depth-d3-mixed-query-branch", "A", "A", 0),
        _sample("q2", "cycle-dir-1-mixed-x-direct-disconnected-l2", "B", "C", 0),
        _sample("q3", "cycle-dir-1-mixed-x-direct-disconnected-l2-control", "D", "D", 0),
    ]
    for tag, predictions in (
        ("v12-4b-1.5k", ("A", "C", "D")),
        ("v12-4b-6k", ("A", "B", "C")),
    ):
        run = tmp_path / tag
        run.mkdir()
        rows = []
        for sample, predicted in zip(cells, predictions):
            row = json.loads(json.dumps(sample))
            row["filtered_resps"] = [predicted]
            rows.append(row)
        (run / f"responses_{SUMMARY.TASK}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    report = SUMMARY.summarize(
        tmp_path, ("v12-4b-1.5k", "v12-4b-6k")
    )

    assert "Controlled proof depth" in report
    assert "Closed-loop condition versus open-chain control" in report
    assert "Semantic subtype by world consistency" in report
    assert "Question family by world consistency" in report
    assert "`direction / closed-loop condition`" in report
    assert "`dir-1 / inconsistent`" in report
    assert "1 fixed by 6k" in report
    assert "1 regressed at 6k" in report


def test_unfiltered_response_parser_only_reads_the_answer_line() -> None:
    sample = _sample(
        "q", "depth-d2-cardinal-none", "B,D", "", 0
    )
    sample.pop("filtered_resps")
    sample["resps"] = ["Reasoning mentions A and E.\nAnswer: B, D"]

    assert SUMMARY.prediction(sample) == ("B", "D")
    assert SUMMARY.strict_correct(sample)


def test_summary_uses_legacy_base_subtype_instead_of_cycle_label() -> None:
    sample = _sample("q", "cycle-which-2-mixed-x-direct-disconnected-l2", "E", "E", 1)
    sample["doc"]["base_semantic_subtype"] = "which-2"
    sample["doc"]["difficulty"]["semantic_subtype"] = "which-cycle"

    assert SUMMARY._semantic_subtype(sample) == "which-2"
    assert SUMMARY._question_family(sample) == "which"
    assert SUMMARY._consistency(sample) == "inconsistent"
