"""Fixed coverage test for controlled recipes, without changing study artifacts."""

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import run_all_controlled_prototype as first
from audit_selected_explanation_depth import replay
from review_unique_pretraining import digest

from spatial.v2.entailment_controlled_coverage import CAP, VERSION, construct, slots
from spatial.v2.entailment_controlled_prototype import check_semantics
from spatial.v2.entailment_dataset import build_attempt
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.entailment_scene_protocol import SHORT_VERSION

ROOT = Path(__file__).resolve().parents[2]


def run_slot(slot):
    result = {"id": slot.identifier, "slot": asdict(slot)}
    try:
        story, problem = construct(slot)
        record = build_attempt(
            story,
            problem,
            slot.requested_depth,
            first.TOKENIZER,
            omission_percent=100 if slot.case == "unlisted" else 0,
            prompt_version=SHORT_VERSION,
        )
        result["record"] = record
        if record["stage"] != "complete":
            raise ValueError(f"{record['status']}: {record.get('reason')}")
        check_semantics(slot, load_evidence(record).semantics)
        rows = [
            r
            for r in record["rows"]
            if r["answer"]
            in (
                "F"
                if slot.case == "unlisted"
                else "ABCD"
                if slot.case == "listed"
                else "E"
            )
        ]
        if len(rows) != 1:
            raise ValueError("requested answer case does not select exactly one menu")
        record["rows"] = rows
        record["tokens"] = [t for t in record["tokens"] if t["id"] == rows[0]["id"]]
        record = replay(record)
        if len(record["tokens"]) != 2:
            raise ValueError("missing matched style measurements")
        if record["rows"][0]["selected_explanation_depth"] != slot.requested_depth:
            raise ValueError("selected certificate depth differs from requested depth")
        if slot.requested_depth is None and record["measurement"]["depth"] is not None:
            raise ValueError("nonpositive E case has a spurious depth")
        result.update(
            record=record,
            status="accepted"
            if all(t["admitted"] for t in record["tokens"])
            else "token-rejected",
            independent_audit="passed",
            sizes=(len(problem.objects), len(problem.premise.operands)),
        )
    except (ValueError, RuntimeError, KeyError, TypeError, AssertionError) as exc:
        result.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    return result


def summarize(results):
    groups = defaultdict(list)
    counts = Counter()
    failures = []
    depth_count_direction = set()
    tokens = []
    for result in results:
        slot = result["slot"]
        counts[slot["kind"], slot["case"], result["status"]] += 1
        if result["status"] != "accepted":
            failures.append({k: v for k, v in result.items() if k != "record"})
            continue
        groups[
            slot["span"], slot["direction"], slot["compact"], slot["extra_size"]
        ].append(result["sizes"])
        tokens.extend(result["record"]["tokens"])
        if (
            slot["kind"] == "count"
            and slot["case"] in ("listed", "unlisted")
            and not slot["compact"]
        ):
            depth_count_direction.add(
                (slot["span"], slot["count_value"], slot["direction"])
            )
    inventory_groups = Counter(s.size_group for s in slots())
    return {
        "attempts": len(results),
        "statuses": dict(Counter(r["status"] for r in results)),
        "count_depth_direction_cells_passed": len(depth_count_direction),
        "count_depth_direction_cells_required": 400,
        "case_counts": [
            {"query": k, "case": c, "status": s, "count": n}
            for (k, c, s), n in sorted(counts.items())
        ],
        "scene_sizes": [
            {
                "span": group[0],
                "direction": group[1],
                "compact": group[2],
                "extra_size": group[3],
                "completed": len(values),
                "expected": inventory_groups[group],
                "sizes": sorted(set(values)),
                "complete_and_matched": len(values) == inventory_groups[group]
                and len(set(values)) == 1,
            }
            for group, values in sorted(groups.items())
        ],
        "max_target_tokens": {
            style: max(
                (t["target_tokens"] for t in tokens if t["style"] == style),
                default=None,
            )
            for style in ("natural", "symbolic")
        },
        "failures": failures,
    }


def write_reports(path, report, results):
    sizes = sorted({tuple(size) for g in report["scene_sizes"] for size in g["sizes"]})
    lines = [
        "# Controlled Coverage Check",
        "",
        "Development verification only. No training or evaluation replacement.",
        "",
        f"- Fixed attempted questions: {report['attempts']}; statuses: {report['statuses']}.",
        f"- Count depth/direction/value cells passed: {report['count_depth_direction_cells_passed']}/400.",
        f"- Original artifact hashes unchanged: {report['originals_unchanged']}.",
        f"- Maximum reference tokens: {report['max_target_tokens']}.",
        f"- Observed (objects, premises): {sizes}.",
        "- All admitted examples have independent spatial evidence replay and matched style references.",
        "- Each question retains one selected menu. F is a sole unlisted entailment, not an E case.",
        "- Positive requested depths are measured, not inferred from sentence count.",
        "- No-entailed and inconsistent E rows receive no positive-depth label.",
        "",
        "## Coverage",
        "",
        "| Query | Case | Status | Questions |",
        "|---|---|---|---|",
    ]
    for item in report["case_counts"]:
        lines.append(
            f"| {item['query']} | {item['case']} | {item['status']} | {item['count']} |"
        )
    lines.extend(
        [
            "",
            "## Scope and Limits",
            "",
            "- Compact scenes have 6 or 9 premises across all applicable answer cases and all eight directions.",
            "- Deep positive coverage crosses L1-L10 and eight directions; Count also crosses values 0-4.",
            "- Each deep positive cell uses one of A-D or F. Both menu roles are separately checked in compact scenes; this is not both roles at every deep cell.",
            "- Multiple-Which cardinalities cycle through 2-4; not every cardinality is crossed with every depth.",
            "- E coverage adds disconnected-component ambiguity and variable strict cycles alongside the original fork and three-edge cycle. It is not exhaustive semantic coverage.",
            "- Deep E examples are checked at every construction span with two direction/recipe variants, not every depth-direction-case product.",
            "- Size matching is a construction check, not proof of absence of all structural shortcuts. No classifier is fitted in this coverage run.",
            "- Seeds are variant groups, not verified canonical families. No canonical deduplication, evaluation-overlap screening, or final 6,000-question selection has been performed.",
            "- The old source quotas and dataset files are untouched. These development examples must not enter training.",
            "- Existing explanation prose is reused; the prior prose-cleanup task is not claimed complete.",
            "",
            f"Full receipt: `{path.name}`",
            "",
            f"Evidence spool: `{report['records_path']}`",
        ]
    )
    if report["failures"]:
        lines.extend(["", "## Failed Slots", ""])
        lines.extend(
            f"- {r['id']}: {r.get('reason', r['status'])}" for r in report["failures"]
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    samples = {}
    for result in results:
        slot = result["slot"]
        if result["status"] == "accepted" and slot["compact"]:
            samples.setdefault(
                (slot["kind"], slot["case"]), result["record"]["rows"][0]
            )
    preview = [
        "# Controlled Compact Examples",
        "",
        "Verified development examples; existing renderers unchanged.",
        "",
    ]
    for (kind, case), row in sorted(samples.items()):
        preview.extend(
            [
                f"## {kind}: {case}",
                "",
                row["prompt"],
                "",
                "### Natural",
                "",
                row["references"]["natural"],
                "",
                "### Symbolic",
                "",
                row["references"]["symbolic"],
                "",
            ]
        )
    path.with_suffix(".examples.md").write_text("\n".join(preview))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    manifest_path = args.report.with_suffix(".manifest.json")
    outputs = [
        args.report,
        manifest_path,
        args.report.with_suffix(".md"),
        args.report.with_suffix(".examples.md"),
    ]
    if any(p.exists() for p in outputs):
        parser.error("existing coverage receipt: do not rerun or overwrite")
    if (
        not args.report.parent.is_dir()
        or not args.tokenizer.is_dir()
        or args.tokenizer.name != first.REVISION
    ):
        parser.error("existing report directory and pinned local tokenizer required")
    first_manifest = json.loads(
        (
            Path(__file__).parent / "ALL-CONTROLLED-PROTOTYPE-v1.manifest.json"
        ).read_text()
    )
    protected = set(first_manifest["original_sha256"]) | set(
        first_manifest["source_sha256"]
    )
    protected.update(
        str(p) for p in Path(__file__).parent.glob("ALL-CONTROLLED-PROTOTYPE-v1*")
    )
    before = {p: digest(Path(p)) for p in sorted(protected)}
    source_files = [
        Path(__file__),
        ROOT / "spatial/v2/entailment_controlled_coverage.py",
        ROOT / "spatial/v2/tests/test_entailment_controlled_coverage.py",
    ]
    source_hashes = {str(p): digest(p) for p in source_files}
    spool = Path(tempfile.mkdtemp(prefix="spatial-controlled-coverage-"))
    records_path = spool / "records.jsonl"
    manifest = {
        "version": VERSION,
        "cap": CAP,
        "slots": [asdict(s) for s in slots()],
        "protected_sha256": before,
        "source_sha256": source_hashes,
        "tokenizer": str(args.tokenizer),
        "workers": args.workers,
        "spool": str(spool),
        "no_refill": True,
        "development_only": True,
    }
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    args.report.write_text(json.dumps({"status": "running", "cap": CAP}) + "\n")
    results = []
    with (
        get_context("spawn").Pool(
            args.workers, initializer=first.initialize, initargs=(str(args.tokenizer),)
        ) as pool,
        records_path.open("x") as stream,
    ):
        for result in pool.imap(run_slot, slots()):
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            results.append(result)
            if len(results) % 80 == 0:
                print(
                    json.dumps(
                        {
                            "completed": len(results),
                            "statuses": dict(Counter(r["status"] for r in results)),
                        }
                    ),
                    flush=True,
                )
    unchanged = all(digest(Path(p)) == h for p, h in before.items())
    sources_unchanged = all(digest(Path(p)) == h for p, h in source_hashes.items())
    report = {
        "status": "coverage-complete-not-training-ready",
        "manifest_sha256": digest(manifest_path),
        "records_path": str(records_path),
        "records_sha256": digest(records_path),
        "originals_unchanged": unchanged,
        "sources_unchanged": sources_unchanged,
        **summarize(results),
    }
    if not unchanged or not sources_unchanged:
        report["status"] = "artifact-change-detected"
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    write_reports(args.report, report, results)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("scene_sizes", "failures", "case_counts")
            }
        )
    )


if __name__ == "__main__":
    main()
