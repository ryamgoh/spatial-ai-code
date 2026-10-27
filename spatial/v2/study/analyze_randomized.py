"""Offline, paired research diagnostics; never rewrite frozen evaluation scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from spatial.v2.evaluation_chat import CHAT_INPUT_PREFIX, encode_chat_input
from spatial.v2.study.metrics import paired_macro_difference
from spatial.v2.study.summarize_randomized import digest, sample_row, tasks_for

WRAPPER = re.compile(
    r"<stage1_response>\n(.*)\n</stage1_response>\n\nAnswer:[^\n]*", re.S
)
FOOTER = re.compile(r"(?:^|\n)Answer:\s*([A-Z](?:\s*,\s*[A-Z])*)\s*$")
BARE = re.compile(r"(?:^|\n)\s*([A-Z](?:\s*,\s*[A-Z])*)\s*$")
SENSITIVE = re.compile(
    r"oracle_letters|oracle_option|answer_certificate|structure_signature|"
    r"hidden_possible_values|<audit>|\"audit\"\s*:"
)


def answer_set(text: str, *, bare: bool = False) -> frozenset[str]:
    match = FOOTER.search(text) or (BARE.search(text) if bare else None)
    letters = tuple(p.strip() for p in match[1].split(",")) if match else ()
    return frozenset(letters) if letters == tuple(sorted(set(letters))) else frozenset()


def normalized_footer(text: str) -> frozenset[str]:
    """Conservative sensitivity check, not a replacement scoring protocol."""
    # Only accept a standalone final answer, never letters inside a rationale.
    cleaned = text.replace("**", "").replace("`", "").strip()
    cleaned = re.sub(r"\s*</think>\s*$", "", cleaned).strip()
    return answer_set(cleaned, bare=True)


def error_type(predicted: frozenset, gold: frozenset) -> str:
    if not predicted:
        return "invalid"
    if predicted == gold:
        return "exact"
    if predicted < gold:
        return "missing_only"
    if predicted > gold:
        return "extra_only"
    return "missing_and_extra"


def source_rows(archive: tarfile.TarFile, member: str):
    with archive.extractfile(member) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def archive_digest(archive: tarfile.TarFile, member: str) -> str:
    with archive.extractfile(member) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def proof_edges(row: dict) -> set[str]:
    profile = row["metadata"]["difficulty"]["proof_complexity"]
    return {"->".join(edge) for edge in profile["ordered_transitions"]}


def audit_sources(benchmark: Path, training: Path, receipt_path: Path, expected: str):
    """Stream archives without extracting; validate hashes before using rows."""
    with tarfile.open(benchmark) as archive:
        member = "prepared-benchmark-candidate/source-natural.jsonl"
        if archive_digest(archive, member) != expected:
            raise ValueError("Benchmark archive does not match evaluation receipt")
        sources = {}
        for row in source_rows(archive, member):
            base = row["metadata"]["base_id"]
            if base in sources:
                raise ValueError("Duplicate benchmark base")
            sources[base] = row
    source_signatures = {r["metadata"]["structure_signature"] for r in sources.values()}
    source_users = {r["messages"][1]["content"] for r in sources.values()}
    receipt = json.loads(receipt_path.read_text())
    report = {"benchmark_sha256": expected, "benchmark_bases": len(sources),
              "benchmark_unique_signatures": len(source_signatures),
              "training_receipt_sha256": digest(receipt_path), "files": {}}
    train_signatures, development_signatures = set(), set()
    with tarfile.open(training) as archive:
        prefix = "prepared-training-candidate/"
        manifest_member = prefix + "manifest.json"
        if archive_digest(archive, manifest_member) != receipt["training_manifest_sha256"]:
            raise ValueError("Training receipt does not bind candidate manifest")
        manifest = json.load(archive.extractfile(manifest_member))
        if manifest["benchmark_jsonl_sha256"] != expected:
            raise ValueError("Training manifest excludes a different benchmark")
        # Manifest uses abbreviations; the proof profile uses full rule labels.
        from spatial.v2.legacy.random_proof import PROOF_RULE_LABELS, deterministic_transition_partition

        heldout = {
            f"{PROOF_RULE_LABELS[a]}->{PROOF_RULE_LABELS[b]}"
            for a, b in deterministic_transition_partition(42).heldout
        }
        report["heldout_transitions"] = sorted(heldout)
        report["benchmark_transition_violations"] = sum(
            (len(proof_edges(r) & heldout) != 1
             if r["metadata"]["benchmark_partition"] == "composition-ood"
             else bool(proof_edges(r) & heldout))
            for r in sources.values()
        )
        for name, info in manifest["files"].items():
            if archive_digest(archive, prefix + name) != info["sha256"]:
                raise ValueError(f"Training source hash mismatch: {name}")
            counts = Counter()
            signatures = set()
            for row in source_rows(archive, prefix + name):
                m = row["metadata"]
                signature = m["structure_signature"]
                signatures.add(signature)
                counts["rows"] += 1
                counts["benchmark_id_overlap"] += m["base_id"] in sources
                counts["benchmark_signature_overlap"] += signature in source_signatures
                counts["benchmark_user_text_overlap"] += row["messages"][1]["content"] in source_users
                counts["heldout_transition_rows"] += bool(proof_edges(row) & heldout)
                counts["sensitive_user_rows"] += bool(SENSITIVE.search(row["messages"][1]["content"]))
            if counts["rows"] != info["rows"]:
                raise ValueError(f"Training source row mismatch: {name}")
            (train_signatures if "/train-" in name else development_signatures).update(signatures)
            report["files"][name] = {**counts, "sha256": info["sha256"],
                                     "unique_signatures": len(signatures)}
    report["train_development_signature_overlap"] = len(train_signatures & development_signatures)
    report["scope"] = (
        "Candidate files bound to a preparation receipt, not proof of cluster consumption. "
        "Signatures and proof profiles are generator-authored; no solver/certificate replay here."
    )
    return sources, report


def inspect_sample(sample: dict, task: dict, source: dict, scorer, token_counter=None) -> dict:
    row = sample_row(sample, task)
    response = sample["filtered_resps"][0]
    match = WRAPPER.fullmatch(response)
    if match is None:
        raise ValueError("Missing native stage-1 wrapper")
    stage1 = match[1]
    replay = scorer(sample["doc"], [response])
    if any(replay[k] != sample[k] for k in replay):
        raise ValueError(f"Scoring replay mismatch: {row['base_id']}")
    if set(source["metadata"]["oracle_letters"]) != set(row["target"].replace(",", "").replace(" ", "")):
        raise ValueError("Saved target differs from hash-bound source")
    prompt, options = sample["arguments"][0]
    messages = source["messages"][:2]
    if prompt.startswith(CHAT_INPUT_PREFIX):
        prompt_matches = prompt == encode_chat_input(messages)
    else:
        expected = (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        prompt_matches = prompt == expected
    gold = frozenset(part.strip() for part in row["target"].split(","))
    pred1, pred2 = answer_set(stage1, bare=True), answer_set(response)
    relaxed = normalized_footer(stage1)
    tokens = token_counter(stage1) if token_counter else None
    return {
        **row, "stage1_pred": ",".join(sorted(pred1)),
        "stage2_pred": ",".join(sorted(pred2)),
        "stage1_size": len(pred1), "stage2_size": len(pred2),
        "stage1_error": error_type(pred1, gold), "stage2_error": error_type(pred2, gold),
        "stage2_missing": len(gold - pred2), "stage2_extra": len(pred2 - gold),
        "normalized_stage1_acc": int(bool(relaxed) and relaxed == gold),
        "normalized_recovery": int(not row["stage1_acc"] and bool(relaxed) and relaxed == gold),
        "prompt_matches_source": int(prompt_matches),
        "sensitive_prompt": int(bool(SENSITIVE.search(prompt))),
        "stage1_chars": len(stage1), "stage1_closed_think": int("</think>" in stage1),
        "stage1_reencoded_tokens": tokens, "stage1_budget": options["max_gen_toks"],
        "near_budget_proxy": int(tokens >= options["max_gen_toks"] - 2) if tokens is not None else None,
        "stage1_tail": stage1[-350:],
    }


def paired(rows: list[dict]) -> list[dict]:
    by_model = defaultdict(dict)
    for row in rows:
        if row["panel"] == "sft":
            by_model[row["model"]][row["base_id"]] = row
    contrasts = []
    for seed in (42, 43):
        contrasts.extend([
            ("untuned", f"multi-natural-seed-{seed}"),
            (f"single-natural-seed-{seed}", f"multi-natural-seed-{seed}"),
            (f"single-answer-only-seed-{seed}", f"single-natural-seed-{seed}"),
        ])
    output = []
    for first, second in contrasts:
        if set(by_model[first]) != set(by_model[second]):
            raise ValueError("Paired models have different base IDs")
        for contract in ("single", "all-possible"):
            cells, a, b = defaultdict(list), {}, {}
            for base, row in by_model[first].items():
                if row["contract"] != contract:
                    continue
                other = by_model[second][base]
                for field in ("target", "contract", "query", "G", "partition", "L", "X"):
                    if row[field] != other[field]:
                        raise ValueError("Paired example metadata mismatch")
                cells[f"{row['query']}/G{row['G']}/{row['partition']}"].append(base)
                a[base], b[base] = bool(row["strict_acc"]), bool(other["strict_acc"])
            result = paired_macro_difference(cells, a, b)
            output.append({
                "first": first, "second": second, "contract": contract,
                "n": len(a), "wins": sum(b[i] and not a[i] for i in a),
                "losses": sum(a[i] and not b[i] for i in a),
                "micro_difference_pp": 100 * sum(int(b[i])-int(a[i]) for i in a) / len(a),
                **result,
            })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot export empty rows")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[d] for d in dimensions)].append(row)
    output = []
    for keys, group in sorted(groups.items()):
        output.append({
            **dict(zip(dimensions, keys)), "n": len(group),
            **{key: sum(r[key] for r in group) / len(group) * 100
               for key in ("strict_acc", "stage1_acc", "normalized_stage1_acc")},
            **{key: sum(r[key] for r in group)
               for key in ("normalized_recovery", "sensitive_prompt")},
            "prompt_mismatches": sum(not r["prompt_matches_source"] for r in group),
            "stage1_invalid": sum(r["stage1_error"] == "invalid" for r in group),
            "stage1_closed_think": sum(r["stage1_closed_think"] for r in group),
            "stage2_singleton": sum(r["stage2_size"] == 1 for r in group),
            **{f"stage2_{kind}": sum(r["stage2_error"] == kind for r in group)
               for kind in ("exact", "missing_only", "extra_only", "missing_and_extra", "invalid")},
            "stage1_correct_to_wrong": sum(r["stage1_acc"] and not r["strict_acc"] for r in group),
            "stage1_wrong_to_correct": sum(not r["stage1_acc"] and r["strict_acc"] for r in group),
            "token_proxy_available": sum(r["near_budget_proxy"] is not None for r in group),
            "near_budget_proxy": sum(r["near_budget_proxy"] or 0 for r in group),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--benchmark-archive", type=Path, required=True)
    parser.add_argument("--training-archive", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--qwen-tokenizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory to preserve previous diagnostics")
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("eval_task_utils", root / "experiments/tasks/utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    sft_receipt = json.loads((args.experiment / "prepared-eval/jobs.json").read_text())
    sources, provenance = audit_sources(
        args.benchmark_archive, args.training_archive, args.training_receipt,
        sft_receipt["benchmark_sha256"],
    )
    tokenizer = None
    if args.qwen_tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.qwen_tokenizer))
        provenance["qwen_tokenizer_sha256"] = digest(args.qwen_tokenizer)
    all_rows, examples, inputs = [], [], {}
    for panel, package in (("sft", "prepared-eval"), ("general", "prepared-general-eval"),
                           ("reasoning", "prepared-reasoning-eval")):
        prepared = args.experiment / package
        receipt = json.loads((prepared / "jobs.json").read_text())
        if receipt["benchmark_sha256"] != sft_receipt["benchmark_sha256"]:
            raise ValueError("Panels evaluate different benchmarks")
        inputs[str(prepared / "jobs.json")] = digest(prepared / "jobs.json")
        for job in receipt["jobs"]:
            name = job["name"]
            folder = prepared / "results" / name
            if not (folder / "COMPLETED").exists():
                raise ValueError(f"Incomplete run: {name}")
            counter = (
                (lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids))
                if tokenizer and (panel == "sft" or name.startswith("qwen3-5-4b-"))
                else None
            )
            seen, example_counts = set(), Counter()
            for task in tasks_for(receipt, name, args.experiment / "prepared-general-eval"):
                path = folder / f"responses_{task['task']}.jsonl"
                inputs[str(path)] = digest(path)
                task_count = 0
                with path.open() as stream:
                    for line_number, line in enumerate(stream, 1):
                        sample = json.loads(line)
                        base = sample["doc"]["metadata"]["base_id"]
                        if base in seen:
                            raise ValueError(f"Duplicate response base: {name}/{base}")
                        seen.add(base)
                        row = inspect_sample(sample, task, sources[base], utils.process_results_v2_answer, counter)
                        row.update(panel=panel, model=name)
                        tail = row.pop("stage1_tail")
                        row.pop("heldout_transition")
                        all_rows.append(row)
                        categories = []
                        if row["normalized_recovery"]:
                            categories.append("normalized_footer_recovers")
                        if row["stage1_acc"] and not row["strict_acc"]:
                            categories.append("stage2_regression")
                        if row["stage1_error"] == "invalid":
                            categories.append("unextractable_stage1")
                        if row["contract"] == "all-possible" and row["stage2_size"] == 1:
                            categories.append("singleton_on_all_possible")
                        for category in categories:
                            if example_counts[category] < 2:
                                examples.append({**row, "category": category, "source": str(path),
                                                 "line": line_number, "stage1_tail": tail})
                                example_counts[category] += 1
                        task_count += 1
                if task_count != task["count"]:
                    raise ValueError("Incomplete response task")
            if seen != set(sources):
                raise ValueError("Model does not cover exactly the frozen benchmark")
            print(f"Audited {panel}/{name}: {len(seen)}", flush=True)
    args.output.mkdir(parents=True)
    for filename, dimensions in (
        ("contracts", ("panel", "model", "contract")),
        ("interface", ("panel", "model")),
        ("partitions", ("panel", "model", "contract", "partition")),
        ("length", ("panel", "model", "contract", "L")),
        ("noise", ("panel", "model", "contract", "X")),
        ("cardinality", ("panel", "model", "contract", "G", "stage2_size")),
        ("query", ("panel", "model", "contract", "query")),
    ):
        write_csv(args.output / f"{filename}.csv", aggregate(all_rows, dimensions))
    write_csv(args.output / "examples.csv", examples)
    for filename, value in (("provenance", provenance), ("paired", paired(all_rows)),
                            ("input_hashes", inputs)):
        (args.output / f"{filename}.json").write_text(json.dumps(value, indent=2) + "\n")
    print(f"Validated {len(all_rows)} rows; diagnostics: {args.output}", flush=True)


if __name__ == "__main__":
    main()
