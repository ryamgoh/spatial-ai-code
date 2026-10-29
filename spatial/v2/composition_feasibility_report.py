"""Recompute descriptive pilot tables; not a release audit or seal."""

import argparse
import json
from collections import Counter
from pathlib import Path

from spatial.v2.composition_contracts import allowed_answers
from spatial.v2.composition_training import digest
from spatial.v2.grading import AnswerMode


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(x) for x in row) + " |" for row in rows),
        ]
    )


def report(candidate):
    manifest = json.loads((candidate / "manifest.json").read_text())
    summary = json.loads((candidate / "summary.json").read_text())
    records = [
        json.loads(line)
        for line in (candidate / "attempts.jsonl").read_text().splitlines()
    ]
    if len(records) != manifest["attempt_budget"]:
        raise ValueError("incomplete pilot; do not report final counts")
    accepted = [r for r in records if r["status"] == "accepted"]
    rows = [row for r in accepted for row in r["data"]["rows"]]
    before = [r for r in records if "data" in r]
    sections = [
        "# Depth Feasibility: Measured Results",
        (
            "Exploratory bounded engineering pilot. Not a training dataset, final "
            "benchmark, completed release audit, or model evaluation."
        ),
        "## Admission",
        table(
            ("Measure", "Count"),
            [
                ("Attempted stories", manifest["stories"]),
                ("Attempted query bases", len(records)),
                *sorted(Counter(r["status"] for r in records).items()),
                ("Admitted views per style", len(rows)),
                ("Admitted stories", len({r["story_id"] for r in accepted})),
                ("Admitted source families", len({r["family_id"] for r in accepted})),
                (
                    "Admitted canonical story groups",
                    len({r["story_key"] for r in accepted}),
                ),
            ],
        ),
        (
            "An admission failure quarantines the entire query base: all menu, "
            "contract and style views. Other queries on that story may remain. "
            "No rejection is replaced with another random draw."
        ),
        "## Controlled Depth",
        (
            "Each cell below starts with six constructed bases. Before-token counts "
            "require evidence verification, actual requested depth, and the "
            "applicable answer-dependence check. These are not six independent "
            "families per depth: source families group deliberately related depth variants."
        ),
        table(
            (
                "Depth",
                "Direction Before/After",
                "Selection Before/After",
                "Count Before/After",
            ),
            [
                [
                    d,
                    *[
                        f"{sum(r['requested_depth'] == d and r['query_kind'] == kind for r in before)} / "
                        f"{sum(r['requested_depth'] == d and r['query_kind'] == kind for r in accepted)}"
                        for kind in ("direction", "which", "count")
                    ],
                ]
                for d in range(1, 11)
            ],
        ),
        "### Admitted Bases by Provisional Role",
        table(
            ("Role / Query / Requested Depth", "Bases"),
            sorted(summary["depth_base_counts"].items()),
        ),
        (
            "Non-controlled random/targeted cases do not fill the table above, even "
            "when a depth descriptor happens to be applicable."
        ),
        "## Gold Mix After Admission",
        table(
            ("Contract", "Views", "E", "F", "E/F Share"),
            [
                [
                    mode.value,
                    len(selected := [r for r in rows if r["contract"] == mode.value]),
                    sum(r["answer"] == "E" for r in selected),
                    sum(r["answer"] == "F" for r in selected),
                    f"{sum(r['answer'] in {'E', 'F'} for r in selected) / len(selected):.1%}"
                    if selected
                    else "n/a",
                ]
                for mode in AnswerMode
            ],
        ),
        (
            "Rates describe this accepted pilot only. Token rejection can bias the "
            "mixture; a lower E/F rate does not certify the sampling policy."
        ),
        table(
            ("Query", "Contract", "Observed Substantive Sizes", "Possible-Set Sizes"),
            [
                [
                    kind,
                    mode.value,
                    ", ".join(
                        map(
                            str,
                            sorted(
                                {
                                    r["returned_count"]
                                    for r in rows
                                    if r["query_kind"] == kind
                                    and r["contract"] == mode.value
                                    and r["returned_count"] is not None
                                }
                            ),
                        )
                    ),
                    ", ".join(
                        map(
                            str,
                            sorted(
                                {
                                    r["possible_count"]
                                    for r in rows
                                    if r["query_kind"] == kind
                                    and r["contract"] == mode.value
                                    and r["semantic_case"] != "inconsistent"
                                }
                            ),
                        )
                    ),
                ]
                for kind in ("direction", "which", "count")
                for mode in AnswerMode
            ],
        ),
        (
            "E/F are escapes, not size-one substantive answers. A count value of "
            "four is one answer value. Possible-set size and returned size are "
            "different measurements. Inapplicable combinations are not quota failures."
        ),
        "### Missing Literal Strings",
        (
            "This is a coverage diagnostic, not a requirement to force all strings "
            "into each query or the determinate extrapolation cohort."
        ),
        table(
            ("Role", "Contract", "Not Observed"),
            [
                [
                    role,
                    mode.value,
                    ", ".join(
                        answer
                        for answer in allowed_answers(mode)
                        if answer
                        not in {
                            r["answer"]
                            for r in rows
                            if r["split"] == role and r["contract"] == mode.value
                        }
                    )
                    or "none",
                ]
                for role in (
                    "train-pilot",
                    "dev-pilot",
                    "in-range-test-pilot",
                    "extrapolation-pilot",
                )
                for mode in AnswerMode
            ],
        ),
        "## Premise-Blind Baseline",
        (
            "For each contract choose its most frequent training-role answer, breaking "
            "ties alphabetically. Apply unchanged to other provisional roles. "
            "These are exploratory pilot diagnostics, not final benchmark scores."
        ),
        table(
            ("Contract", "Training-Selected Prediction"),
            [
                (mode, item["prediction_from_train_only"])
                for mode, item in summary["baseline"].items()
            ],
        ),
        table(
            ("Role", "Correct", "Total", "Accuracy"),
            [
                [
                    role,
                    correct := sum(
                        item["roles"][role]["correct"]
                        for item in summary["baseline"].values()
                    ),
                    total := sum(
                        item["roles"][role]["total"]
                        for item in summary["baseline"].values()
                    ),
                    f"{correct / total:.1%}" if total else "n/a",
                ]
                for role in (
                    "train-pilot",
                    "dev-pilot",
                    "in-range-test-pilot",
                    "extrapolation-pilot",
                )
            ],
        ),
        "## Token Maxima Before Admission",
        table(
            ("Style", "Target", "Full Training", "Stage-2 Reservation"),
            [
                (
                    style,
                    values["target_tokens"],
                    values["training_tokens"],
                    values["stage2_reserved_tokens"],
                )
                for style, values in summary["token_maxima_before_admission"].items()
            ],
        ),
        (
            "Pinned Qwen3.5-4B tokenizer; Stage-1 cap 4,096, context 8,192, "
            "Stage-2 cap 16. No truncation, model loading or GPU inference."
        ),
        "## Structural Scope",
        table(
            (
                "Cohort",
                "Stories",
                "Canonical Groups",
                "Degree Profiles",
                "Object Range",
            ),
            [
                [
                    cohort,
                    len(
                        selected := {
                            r["story_id"]: r
                            for r in before
                            if (r["requested_depth"] is not None) == controlled
                        }
                    ),
                    len({r["story_key"] for r in selected.values()}),
                    len(
                        {
                            tuple(r["data"]["graph"]["degree_sequence"])
                            for r in selected.values()
                        }
                    ),
                    (
                        f"{min((r['data']['graph']['objects'] for r in selected.values()), default=0)}-"
                        f"{max((r['data']['graph']['objects'] for r in selected.values()), default=0)}"
                    ),
                ]
                for cohort, controlled in (
                    ("controlled", True),
                    ("random/targeted", False),
                )
            ],
        ),
        (
            "Degree profiles and canonical identities are descriptive, not proof of "
            "rich structural diversity. The controlled constructor uses one backbone "
            "with optional leaf branches and disconnected distractors. It does not "
            "yet randomize independent x/y networks, multiple essential deep members, "
            "or general query-relevant branching. Gold labels can correlate with "
            "these scaffolds; shortcut baselines beyond contract-majority remain needed."
        ),
        "## Rejections",
        table(
            ("Story", "Query", "Status", "Reason"),
            [
                (
                    r["story_id"],
                    r["query_kind"],
                    r["status"],
                    r.get("reason")
                    or "; ".join(
                        f"{t['style']} {t['id']}: {t['supervised_tokens']} supervised tokens"
                        for t in r.get("data", {}).get("tokens", [])
                        if not t["admitted"]
                    ),
                )
                for r in records
                if r["status"] != "accepted"
            ],
        ),
        "## Reproduction",
        (
            "Run `python -m spatial.v2.composition_feasibility` with the pinned "
            "tokenizer and a new output directory using the dependencies recorded "
            "in `manifest.json`. The fixed pre-run schedule and every attempted "
            "problem are preserved. Run `python -m spatial.v2.composition_feasibility_report "
            "--candidate <directory> --output <report.md>` to recompute these tables."
        ),
        (
            "The manifest records generation source and tokenizer hashes; it is "
            "not a dataset seal. No final training files or benchmark release were created."
        ),
    ]
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.samples and args.samples.exists()):
        raise ValueError("refusing to overwrite a report")
    args.output.write_text(report(args.candidate))
    if args.samples:
        selected = {
            ("depth-1-sample-0", "direction"),
            ("depth-1-sample-0", "which"),
            ("depth-1-sample-0", "count"),
            ("depth-6-sample-0", "count"),
            ("depth-10-sample-0", "direction"),
            ("depth-10-sample-1", "count"),
        }
        sections = [
            "# Feasibility Paired Examples",
            (
                "Six selected pairs, not a complete category/depth release audit. "
                "The depth-ten Direction base was token-rejected because another "
                "contract view is too long; its compact Unique view is shown here."
            ),
        ]
        for line in (args.candidate / "attempts.jsonl").read_text().splitlines():
            record = json.loads(line)
            if (record["story_id"], record["query_kind"]) not in selected:
                continue
            row = record["data"]["rows"][0]
            sections.extend(
                [
                    f"## {row['id']}",
                    f"Base status: {record['status']}; measured depth: {row['depth']}.",
                    "### Prompt",
                    row["prompt"],
                    "### Natural",
                    row["references"]["natural"],
                    "### Symbolic",
                    row["references"]["symbolic"],
                ]
            )
        args.samples.write_text("\n\n".join(sections) + "\n")
    print(f"Wrote {args.output}; SHA256 {digest(args.output)}")


if __name__ == "__main__":
    main()
