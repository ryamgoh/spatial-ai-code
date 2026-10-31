"""Summarize unsealed candidate exposure and export paired review samples."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from spatial.v2.composition_training import digest

SPLITS = ("train", "dev", "in-range-test", "extrapolation-test")
CONTRACTS = ("unique-entailed", "listed-entailed", "exhaustive-entailed")


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(map(str, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.candidate / "manifest.json").read_text())
    if manifest["status"] != "completed-unsealed-candidate" or manifest["sealed"]:
        raise ValueError("review requires a completed unsealed candidate")
    bases, views, original_views, augmented, sources = Counter(), Counter(), Counter(), Counter(), Counter()
    escapes, originals = defaultdict(Counter), defaultdict(Counter)
    lengths, depth_lengths, samples = defaultdict(list), defaultdict(list), {}
    with (args.candidate / "attempts.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            split, kind = record["split"], record["query_kind"]
            bases[split, kind] += 1
            sources[split, record["source"]] += 1
            augmented[split] += any(r.get("position_variant", 0) > 0 for r in record["rows"])
            for token in record["tokens"]:
                lengths[token["style"]].append(token["target_tokens"])
                depth_lengths[record["rows"][0]["depth"], token["style"]].append(token["target_tokens"])
            for row in record["rows"]:
                key = (split, row["contract"])
                views[split] += 1
                escapes[key][row["answer"]] += 1
                if not row.get("position_variant", 0):
                    original_views[split] += 1
                    originals[key][row["answer"]] += 1
                categories = [
                    f"query-{kind}", f"contract-{row['contract']}",
                    f"depth-{row.get('depth')}", f"escape-{row.get('escape_reason')}",
                ]
                if row["menu_role"] == "omission":
                    categories.append("omission")
                if row.get("sole_entailed_with_contingent"):
                    categories.append("sole-entailed-with-contingent")
                if row.get("position_variant", 0):
                    categories.append("position-variant")
                if len(record["semantics"]["entailed"]) == 3:
                    categories.append("three-entailed")
                for category in categories:
                    if category not in samples:
                        samples[category] = row
    base_table = [
        [split, *(bases[split, kind] for kind in ("direction", "which", "count")),
         sum(bases[split, kind] for kind in ("direction", "which", "count"))]
        for split in SPLITS
    ]
    exposure_table = [
        [split, original_views[split], views[split], augmented[split],
         views[split] - original_views[split]]
        for split in SPLITS
    ]
    escape_table = []
    for split in SPLITS:
        for contract in CONTRACTS:
            key = split, contract
            before, after = originals[key], escapes[key]
            escape_table.append([
                split, contract,
                f"{100 * (before['E'] + before['F']) / sum(before.values()):.2f}%",
                f"{100 * (after['E'] + after['F']) / sum(after.values()):.2f}%",
                after["E"], after["F"], sum(after.values()),
            ])
    report = [
        "# Entailment Candidate Distribution Review",
        "",
        "Status: unsealed candidate. This report is descriptive, not approval to train.",
        f"Candidate manifest SHA-256: `{digest(args.candidate / 'manifest.json')}`.",
        "",
        "## Retained Bases",
        table(["Split", "Direction", "Selection", "Count", "Total"], base_table),
        "",
        "## Correlated View Exposure",
        table(["Split", "Original Views/Style", "Final Views/Style",
               "Augmented Bases", "Added Views/Style"], exposure_table),
        "",
        "Position views are not additional independent bases or families. Both styles",
        "receive identical questions, menus and labels. Report base-level or",
        "family-clustered evaluation uncertainty, not independence of all views.",
        "Multi-entailed Selection receives more exposure under this augmentation.",
        "",
        "## Escape Distribution",
        table(["Split", "Contract", "Original E/F", "Final E/F", "E", "F", "Final Views"],
              escape_table),
        "",
        "Original means position variant zero, including its omission companion.",
        "E/F prevalence alone is not a correctness or difficulty measure.",
        "",
        "## Sources",
        table(["Split", "Controlled", "Random World", "Random Atoms", "Targeted"], [
            [split, *(sources[split, source] for source in
                      ("controlled", "random-world", "random-atoms", "targeted"))]
            for split in SPLITS
        ]),
        "",
        "## Explanation Lengths",
        table(["Style", "Min Tokens", "Mean Tokens", "Max Tokens"], [
            [style, min(values), f"{mean(values):.2f}", max(values)]
            for style, values in sorted(lengths.items())
        ]),
        "",
        "Lengths use pinned-tokenizer measurements saved by full assembly.",
        "",
        table(["Depth", "Style", "Min Tokens", "Mean Tokens", "Max Tokens"], [
            [depth, style, min(values), f"{mean(values):.2f}", max(values)]
            for (depth, style), values in sorted(
                depth_lengths.items(), key=lambda item: (item[0][0] or 0, item[0][1])
            )
        ]),
        "",
        "## Review Status",
        "Automated quota and coverage checks are recorded in `summary.json`.",
        "Independent evidence/input audit and human-readable paired sample review",
        "must be recorded separately. Sample export is not itself a manual pass.",
        "No seal or training authorization is granted by this report.",
        "",
    ]
    args.report.write_text("\n".join(report))
    output = ["# Paired Reference Review Samples", "",
              "Deterministic first occurrence per review category; not a random quality estimate.", ""]
    emitted = set()
    for category, row in samples.items():
        if row["id"] in emitted:
            continue
        emitted.add(row["id"])
        covered = [c for c, r in samples.items() if r["id"] == row["id"]]
        output.extend([
            f"## {row['id']}", "", f"Categories: {', '.join(covered)}.",
            f"Split: {row['split']}; contract: {row['contract']}; answer: {row['answer']}.",
            "", "### Question", "", row["prompt"], "",
            "### Natural", "", row["references"]["natural"], "",
            "### Symbolic", "", row["references"]["symbolic"], "",
        ])
    args.samples.write_text("\n".join(output))
    print(json.dumps({"bases": sum(bases.values()), "views_per_style": sum(views.values()),
                      "paired_review_samples": len(emitted)}, indent=2))


if __name__ == "__main__":
    main()
