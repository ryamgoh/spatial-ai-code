"""One fixed 260-slot experiment; never select, replace, seal, or train a dataset."""

import argparse
import json
import tempfile
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

from audit_selected_explanation_depth import replay
from review_unique_pretraining import digest, features

from spatial.v2.entailment_controlled_prototype import (
    VERSION,
    check_semantics,
    construct,
    slots,
)
from spatial.v2.entailment_dataset import build_attempt
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.entailment_scene_protocol import SHORT_VERSION

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = None


def initialize(path):
    global TOKENIZER
    from transformers import AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained(path, local_files_only=True)


def run_slot(slot):
    result = {"slot": asdict(slot), "id": slot.identifier}
    try:
        story, problem = construct(slot)
        record = build_attempt(
            story,
            problem,
            slot.requested_depth,
            TOKENIZER,
            omission_percent=100 if slot.case == "unlisted" else 0,
            prompt_version=SHORT_VERSION,
        )
        result["record"] = record
        if record["stage"] != "complete":
            raise ValueError(f"{record['status']}: {record.get('reason')}")
        evidence = load_evidence(record)
        check_semantics(slot, evidence.semantics)
        wanted = [
            row
            for row in record["rows"]
            if (
                row["answer"] == "F"
                if slot.case == "unlisted"
                else row["answer"] in "ABCD"
                if slot.case == "listed"
                else row["answer"] == "E"
            )
        ]
        if len(wanted) != 1:
            raise ValueError("expected exactly one menu for the requested answer case")
        record["rows"] = wanted
        record["tokens"] = [t for t in record["tokens"] if t["id"] == wanted[0]["id"]]
        if len(record["tokens"]) != 2:
            raise ValueError("missing matched style token measurements")
        record = replay(record)
        row = record["rows"][0]
        if row["selected_explanation_depth"] != slot.requested_depth:
            raise ValueError("selected explanation has the wrong actual proof depth")
        if slot.requested_depth is None and record["measurement"]["depth"] is not None:
            raise ValueError("ambiguous or inconsistent E acquired a positive depth")
        record["status"] = (
            "accepted"
            if all(t["admitted"] for t in record["tokens"])
            else "token-rejected"
        )
        result.update(
            record=record,
            status=record["status"],
            independent_audit="passed",
            features=features(record),
            answer=row["answer"],
            selected_depth=row["selected_explanation_depth"],
        )
    except (ValueError, RuntimeError, KeyError, TypeError, AssertionError) as exc:
        # Every failed scheduled slot is reported; never retry or refill it.
        result.update(
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    return result


def probes(results):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.tree import DecisionTreeClassifier

    reports = []
    for kind in ("direction", "which", "count"):
        subset = [
            r
            for r in results
            if r["slot"]["kind"] == kind and r.get("independent_audit") == "passed"
        ]
        if len(subset) != (100 if kind == "which" else 80):
            reports.append({"query": kind, "status": "incomplete-no-probe"})
            continue
        x = np.array([r["features"] for r in subset])
        y = np.array([r["answer"] == "E" for r in subset], dtype=int)
        groups = np.array([r["record"]["family_id"] for r in subset])
        for name, columns, depth in (
            ("size_only", 2, 1),
            ("structural_counts", 13, 3),
        ):
            predicted = np.zeros(len(y), dtype=int)
            for train, test in GroupKFold(n_splits=5).split(x, y, groups):
                if set(groups[train]) & set(groups[test]):
                    raise ValueError("prototype family leaked across probe folds")
                tree = DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=31031,
                )
                tree.fit(x[train, :columns], y[train])
                predicted[test] = tree.predict(x[test, :columns])
            reports.append(
                {
                    "query": kind,
                    "model": name,
                    "accuracy": float(accuracy_score(y, predicted)),
                    "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
                    "majority_accuracy": float(max(y.mean(), 1 - y.mean())),
                    "families": len(set(groups)),
                }
            )
    return reports


def summarize(results):
    sizes, cases, failures, tokens = defaultdict(list), Counter(), [], []
    for r in results:
        slot = r["slot"]
        cases[slot["kind"], slot["case"], r["status"]] += 1
        if r["status"] == "failed":
            failures.append({k: r[k] for k in ("id", "reason", "traceback")})
            continue
        sizes[slot["span"], slot["extra_size"]].append(tuple(r["features"][:2]))
        tokens.extend(r["record"]["tokens"])
    return {
        "statuses": dict(Counter(r["status"] for r in results)),
        "case_counts": [
            {"query": k, "case": c, "status": s, "count": n}
            for (k, c, s), n in sorted(cases.items())
        ],
        "size_groups": [
            {
                "construction_span": span,
                "extra_size": extra,
                "completed_slots": len(values),
                "objects_and_premises": sorted(set(values)),
                "all_13_cases_matched": len(values) == 13 and len(set(values)) == 1,
            }
            for (span, extra), values in sorted(sizes.items())
        ],
        "max_target_tokens": {
            style: max(
                (t["target_tokens"] for t in tokens if t["style"] == style),
                default=None,
            )
            for style in ("natural", "symbolic")
        },
        "probes": probes(results),
        "failures": failures,
    }


def markdown(report):
    lines = [
        "# All-Controlled Prototype",
        "",
        "Separate feasibility experiment, not a replacement training dataset.",
        "",
        f"- Fixed attempts: {report['attempts']}; statuses: {report['statuses']}.",
        "- All ten construction spans; two matched scene sizes per span.",
        (
            "- L denotes verified positive certificate depth only. Ambiguous and "
            "inconsistent E rows have no positive L label."
        ),
        (
            "- Count sole values cycle through 0-4. This is not full depth-by-value "
            "coverage or the final 1,200/600/200 allocation."
        ),
        "- Existing Natural/Symbolic renderers are reused without prose changes.",
        (
            "- Every completed case is independently replayed, including "
            "counterexamples and positive/conflict certificates."
        ),
        f"- Maximum target tokens: {report['max_target_tokens']}.",
        f"- Original artifact fingerprints unchanged: {report['originals_unchanged']}.",
        "",
        "## Scene Sizes",
        "",
        "| Construction span | Size variant | Objects / premises | Cases matched |",
        "|---|---|---|---|",
    ]
    for group in report["size_groups"]:
        lines.append(
            f"| {group['construction_span']} | {group['extra_size']} | "
            f"{group['objects_and_premises']} | {group['all_13_cases_matched']} |"
        )
    lines.extend(
        [
            "",
            "## Shortcut Checks",
            "",
            (
                "Five grouped folds, keeping all variants of each seed/span together. "
                "Ten construction families only; no SFT model or evaluation examples. "
                "The structural tree uses min_samples_leaf=5 because this prototype "
                "is much smaller than the previous training-set diagnostic (25)."
            ),
            "",
            "| Query | Probe | Accuracy | Balanced accuracy |",
            "|---|---|---|---|",
        ]
    )
    for probe in report["probes"]:
        if "model" in probe:
            lines.append(
                f"| {probe['query']} | {probe['model']} | "
                f"{probe['accuracy']:.1%} | {probe['balanced_accuracy']:.1%} |"
            )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            (
                "- Exact object/premise-count matching does not establish absence "
                "of graph, wording, semantic-recipe, or explanation shortcuts."
            ),
            (
                "- The current ambiguity recipe uses an unordered pair above an "
                "anchor; the inconsistency recipe uses a three-edge strict cycle. "
                "These are narrow recognizable mechanisms, not broad E coverage."
            ),
            (
                "- Context is constructed on an excluded side of the reference. "
                "Every context object participates in Which/Count verification, "
                "but this is a restricted family of scenes."
            ),
            (
                "- No canonical deduplication, final family-floor check, held-out "
                "overlap screening, or production allocation was performed. All "
                "prototype rows are development-only and must not enter training."
            ),
            (
                "- Prototype case proportions differ from the final study; these "
                "diagnostic percentages are not directly comparable to the old "
                "70%-non-E training distribution."
            ),
            (
                "- No full replacement, source-quota migration, evaluation changes, "
                "sealing, training, or automatic extra sampling."
            ),
            "",
            f"Manifest: `{report['manifest_path']}`",
            "",
            f"Full evidence and references: `{report['records_path']}`",
        ]
    )
    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {r['id']}: {r['reason']}" for r in report["failures"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    manifest_path = args.report.with_suffix(".manifest.json")
    if any(
        p.exists() for p in (args.report, manifest_path, args.report.with_suffix(".md"))
    ):
        parser.error("existing prototype receipt: do not rerun or overwrite")
    if not args.report.parent.is_dir() or args.tokenizer.name != REVISION:
        parser.error("use an existing report directory and the pinned tokenizer")
    inventory = slots()
    if len(inventory) != 260:
        raise ValueError("prototype cap changed")
    original_files = [
        p
        for directory in ("unique-training-candidate-v1", "spatialmap-unique-v1")
        for p in sorted((Path(__file__).parent / directory).iterdir())
        if p.suffix == ".jsonl" or p.name == "manifest.json"
    ]
    original_files.extend(
        Path(__file__).parent / name
        for name in (
            "entailment-config-frozen-v1.json",
            "unique-only-selection-v1/selection.json",
            "UNIQUE-TRAINING-SELECTION-REPLAY-v1.json",
        )
    )
    original_hashes = {str(p): digest(p) for p in original_files}
    spool = Path(tempfile.mkdtemp(prefix="spatial-all-controlled-prototype-"))
    records_path = spool / "records.jsonl"
    sources = [
        Path(__file__),
        Path(__file__).parent / "audit_entailment_candidate.py",
        Path(__file__).parent / "audit_selected_explanation_depth.py",
        Path(__file__).parent / "review_unique_pretraining.py",
        *[
            ROOT / "spatial/v2" / name
            for name in (
                "entailment_controlled_prototype.py",
                "composition_scene_sampler.py",
                "entailment_dataset.py",
                "entailment_evidence.py",
                "entailment_contracts.py",
                "entailment_scene_protocol.py",
                "entailment_names.py",
                "solver.py",
            )
        ],
    ]
    manifest = {
        "version": VERSION,
        "cap": 260,
        "slots": [asdict(s) for s in inventory],
        "source_sha256": {str(p): digest(p) for p in sources},
        "original_sha256": original_hashes,
        "tokenizer": str(args.tokenizer),
        "workers": args.workers,
        "spool": str(spool),
        "no_refill": True,
        "development_only": True,
        "packages": {
            p: version(p) for p in ("z3-solver", "transformers", "scikit-learn")
        },
    }
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    args.report.write_text(json.dumps({"status": "running", "cap": 260}) + "\n")
    results = []
    with (
        get_context("spawn").Pool(
            args.workers, initializer=initialize, initargs=(str(args.tokenizer),)
        ) as pool,
        records_path.open("x") as stream,
    ):
        for result in pool.imap(run_slot, inventory):
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            results.append(result)
            if len(results) % 26 == 0:
                print(
                    json.dumps(
                        {
                            "completed": len(results),
                            "statuses": dict(Counter(r["status"] for r in results)),
                        }
                    ),
                    flush=True,
                )
    unchanged = all(digest(Path(p)) == h for p, h in original_hashes.items())
    report = {
        "status": "prototype-complete-not-a-training-release",
        "attempts": len(results),
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "records_path": str(records_path),
        "records_sha256": digest(records_path),
        "originals_unchanged": unchanged,
        **summarize(results),
    }
    if not unchanged:
        report["status"] = "original-artifact-change-detected"
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    args.report.with_suffix(".md").write_text(markdown(report))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("failures", "size_groups")}
        )
    )


if __name__ == "__main__":
    main()
