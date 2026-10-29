"""Pinned token measurement, matched training exports, and fail-closed sealing."""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import yaml

from spatial.v2.composition_contracts import parse_answer, parse_prompt
from spatial.v2.composition_pilot import check_splits

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
CONTEXT = 8192
STAGE1_CAP = 4096
STAGE2_CAP = 16
STYLES = {
    "natural": (
        "Solve the spatial problem using the stated map assumptions and answer contract. "
        "Explain the necessary deductions or witness maps in ordinary English, "
        "including coordinate comparisons where useful. Cite premises by their "
        "1-based sentence positions when needed. End with exactly one line "
        "'Answer: <letters>' using the required letter format."
    ),
    "symbolic": (
        "Solve the spatial problem using the stated map assumptions and answer contract. "
        "Explain the necessary deductions or witness maps using direction predicates "
        "and explicit x/y comparisons. Cite premises by their 1-based sentence "
        "positions when needed. End with exactly one line "
        "'Answer: <letters>' using the required letter format."
    ),
}
COMPLETION = (
    "Based on the question and your reasoning above, give only the final answer."
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def messages(row, style):
    return [
        {"role": "system", "content": STYLES[style]},
        {"role": "user", "content": row["prompt"]},
    ]


def prompt_text(tokenizer, row, style):
    return tokenizer.apply_chat_template(
        messages(row, style),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def completion_suffix(*, terminated=False):
    # Explicit raw replay avoids Qwen's multi-turn template removing <think>
    # content from a preceding assistant turn.
    end = "" if terminated else "<|im_end|>"
    return f"{end}\n<|im_start|>user\n{COMPLETION}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def tokenize_example(tokenizer, row, style):
    source = messages(row, style)
    target = row["references"][style]
    if parse_answer(target, row["contract"], stage=1) != row["answer"]:
        raise ValueError("invalid training answer")
    prefix = prompt_text(tokenizer, row, style)
    full = tokenizer.apply_chat_template(
        [*source, {"role": "assistant", "content": target}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if full != prefix + target.strip() + "<|im_end|>\n":
        raise ValueError("training and inference templates disagree")
    ids = tokenizer.encode(full, add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("token boundary crosses the assistant loss mask")
    labels = [-100] * len(prefix_ids) + ids[len(prefix_ids) :]
    if len(ids) > CONTEXT or len(ids) - len(prefix_ids) > STAGE1_CAP:
        raise ValueError(f"token budget exceeded: {row['id']}/{style}")
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
    }, {
        "id": row["id"],
        "style": style,
        "split": row["split"],
        "prompt_tokens": len(prefix_ids),
        "training_tokens": len(ids),
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "supervised_tokens": len(ids) - len(prefix_ids),
        "stage2_reserved_tokens": len(prefix_ids)
        + STAGE1_CAP
        + len(tokenizer.encode(completion_suffix(), add_special_tokens=False))
        + STAGE2_CAP,
    }


def evidence_categories(row, base):
    sem, kind = row["semantic_case"], row["query_kind"]
    categories = []
    if sem == "inconsistent":
        categories.append(base["evidence"]["certificate"]["kind"])
    elif kind == "direction":
        categories.append(
            "direction-proof" if sem == "unique" else "direction-ambiguity"
        )
    elif kind == "which":
        categories.append("selection-" + sem)
    elif sem == "unique":
        possible = base["evidence"]["semantics"]["possible"]
        categories.append("count-zero" if possible == [0] else "count-invariant")
    else:
        categories.append("count-joint-variable")
    if row["recipe"] == "cycle" and sem != "inconsistent":
        categories.append("consistent-cycle")
    if row["escape_reason"] in {"incomplete-coverage", "unique-unlisted"}:
        categories.append("missing-menu")
    return categories


def audit_selection(pack):
    chosen, cells, categories = set(), {}, defaultdict(dict)
    for row in pack["rows"]:
        cell = "/".join(row[k] for k in ("query_kind", "contract", "semantic_case"))
        cells.setdefault(cell, row["id"])
        for category in evidence_categories(row, pack["bases"][row["base_id"]]):
            # Different questions about the same story do not count as five bases.
            categories[category].setdefault(row["story_id"], row["id"])
    chosen.update(cells.values())
    for category, stories in categories.items():
        if len(stories) < 5:
            raise ValueError(
                f"fewer than five distinct stories for evidence category {category}"
            )
        chosen.update(list(stories.values())[:5])
    return {
        "sample_ids": sorted(chosen),
        "cells": cells,
        "categories": {k: list(v.values())[:5] for k, v in categories.items()},
    }


def validate_pack(pack):
    rows = pack["rows"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty data or duplicate view IDs")
    if any(a["status"] == "verification-failed" for a in pack["attempts"]):
        raise ValueError("unresolved verification failures block preparation")
    check_splits(rows, pack["bases"])
    expected = {
        f"{kind} / {contract} / {case}"
        for kind, cases in (
            ("direction", ("unique", "underdetermined", "inconsistent")),
            ("count", ("unique", "underdetermined", "inconsistent")),
            (
                "which",
                (
                    "unique",
                    "no-match",
                    "multiple-certain-match",
                    "underdetermined",
                    "inconsistent",
                ),
            ),
        )
        for contract in ("single", "visible-possible", "all-possible")
        for case in cases
    }
    for split in ("train", "dev", "test"):
        actual = {
            f"{r['query_kind']} / {r['contract']} / {r['semantic_case']}"
            for r in rows
            if r["split"] == split
        }
        if actual != expected:
            raise ValueError(
                f"{split} missing required cells: {sorted(expected - actual)}"
            )
    for row in rows:
        _, menu, contract = parse_prompt(row["prompt"])
        if tuple(row["menu"]) != menu or contract.value != row["contract"]:
            raise ValueError("stored menu/contract mismatch")
        for style in STYLES:
            if (
                parse_answer(row["references"][style], contract, stage=1)
                != row["answer"]
            ):
                raise ValueError("stored explanation answer mismatch")
    return audit_selection(pack)


def distribution(values):
    values = sorted(values)
    return {
        "min": values[0],
        "median": values[math.ceil(len(values) * 0.5) - 1],
        "p95": values[math.ceil(len(values) * 0.95) - 1],
        "max": values[-1],
        "total": sum(values),
    }


def training_config(style, prefix):
    return {
        "base_model": MODEL,
        "revision_of_model": REVISION,
        "adapter": "qlora",
        "load_in_4bit": True,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.out_proj",
        ],
        "lora_qkv_kernel": False,
        "lora_o_kernel": False,
        "lora_mlp_kernel": False,
        "embeddings_skip_upcast": True,
        "attn_implementation": "sdpa",
        "sample_packing": False,
        "pad_to_sequence_len": False,
        "group_by_length": False,
        "shuffle_merged_datasets": False,
        "train_on_inputs": False,
        "learning_rate": 5e-5,
        "sequence_len": CONTEXT,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optimizer": "paged_adamw_8bit",
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "fp16": False,
        "bf16": True,
        "max_grad_norm": 0.05,
        "num_epochs": 2,
        "logging_steps": 1,
        "output_dir": f"{prefix}/models/{style}-seed-42",
        "datasets": [
            {
                "path": f"{prefix}/data/{style}-train-tokenized.jsonl",
                "ds_type": "json",
                "type": None,
            }
        ],
        "dataset_prepared_path": f"{prefix}/cache/{style}",
        "val_set_size": 0,
        "eval_strategy": "no",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": False,
        "seed": 42,
        "data_seed": 42,
        "dataset_num_proc": 1,
        "dataloader_num_workers": 0,
        "use_wandb": False,
    }


def prepare(candidate, output, tokenizer_dir):
    candidate, output, tokenizer_dir = (
        Path(candidate),
        Path(output),
        Path(tokenizer_dir),
    )
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite a prepared package")
    receipt = json.loads((candidate / "receipt.json").read_text())
    root = Path(__file__).resolve().parents[2]
    for name, expected in receipt["output_sha256"].items():
        if digest(candidate / name) != expected:
            raise ValueError(f"candidate changed: {name}")
    for name, expected in receipt["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(
                f"candidate source changed: {name}; regenerate before sealing"
            )
    pack = json.loads((candidate / "pilot.json").read_text())
    selected = validate_pack(pack)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    (output / "configs").mkdir(exist_ok=True)
    order = sorted(pack["rows"], key=lambda r: r["id"])
    random.Random(42).shuffle(order)
    token_rows = []
    for style in STYLES:
        inputs = {split: [] for split in ("train", "dev", "test")}
        train, train_messages = [], []
        for row in order:
            tokenized, measurement = tokenize_example(tokenizer, row, style)
            if measurement["stage2_reserved_tokens"] > CONTEXT:
                raise ValueError("Stage-2 context reserve exceeds context capacity")
            token_rows.append(measurement)
            inputs[row["split"]].append(
                {
                    "id": row["id"],
                    "contract": row["contract"],
                    "messages": messages(row, style),
                }
            )
            if row["split"] == "train":
                train.append(tokenized)
                train_messages.append(
                    {
                        "messages": [
                            *messages(row, style),
                            {"role": "assistant", "content": row["references"][style]},
                        ]
                    }
                )
        for split, data in inputs.items():
            write_jsonl(output / "data" / f"{style}-{split}-inputs.jsonl", data)
        write_jsonl(output / "data" / f"{style}-train-tokenized.jsonl", train)
        write_jsonl(output / "data" / f"{style}-train-messages.jsonl", train_messages)
        prefix = str(output.resolve().relative_to(root))
        (output / "configs" / f"{style}.yaml").write_text(
            yaml.safe_dump(training_config(style, prefix), sort_keys=False)
        )
    write_jsonl(output / "token-measurements.jsonl", token_rows)
    summary = {
        style: {
            split: {
                key: distribution(
                    [
                        r[key]
                        for r in token_rows
                        if r["style"] == style and r["split"] == split
                    ]
                )
                for key in (
                    "prompt_tokens",
                    "target_tokens",
                    "supervised_tokens",
                    "training_tokens",
                    "stage2_reserved_tokens",
                )
            }
            for split in ("train", "dev", "test")
        }
        for style in STYLES
    }
    (output / "token-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    selected["rows_sha256"] = {
        row["id"]: content_hash(row)
        for row in pack["rows"]
        if row["id"] in selected["sample_ids"]
    }
    (output / "audit-selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    # An input-only dataset is separate from the protected labels and reference bank.
    write_jsonl(output / "references.jsonl", order)
    manifest = {
        "status": "prepared-awaiting-audit-seal",
        "candidate_sha256": digest(candidate / "pilot.json"),
        "candidate_path": str(candidate.resolve().relative_to(root)),
        "model": MODEL,
        "revision": REVISION,
        "seed": 42,
        "counts": pack["counts"],
        "attempt_policy": pack["config"],
        "tokenizer_sha256": {
            name: digest(tokenizer_dir / name)
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            )
        },
        "preparation_versions": {
            name: version(name) for name in ("transformers", "tokenizers", "jinja2")
        },
        "style_prompts": STYLES,
        "context": CONTEXT,
        "stage1_cap": STAGE1_CAP,
        "stage2_cap": STAGE2_CAP,
        "enable_thinking": False,
        "selection": "macro-contract Stage-1 dev accuracy; earliest epoch wins ties",
        "accuracy_loss_tolerance_pp": 2,
        "epochs": 2,
        "evaluation": {
            "temperature": 0,
            "top_p": 1,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "stage2_all_rows": True,
        },
        "audit_selection_sha256": digest(output / "audit-selection.json"),
        "source_sha256": {
            str(p.relative_to(root)): digest(p)
            for p in sorted((root / "spatial/v2").glob("*.py"))
        },
        "environment_sha256": {
            name: digest(root / name)
            for name in ("pyproject.toml", "uv.lock", "finetune/finetune.py")
        },
        "files": {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
            and p.name not in {"manifest.json", "seal.json", "audit-review.json"}
            and "models" not in p.parts
            and "cache" not in p.parts
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "counts": manifest["counts"],
                "audit_pairs": len(selected["sample_ids"]),
            },
            indent=2,
        )
    )
    return manifest


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def verify_prepared(output):
    output = Path(output)
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(output / name) != expected:
            raise ValueError(f"prepared file changed: {name}")
    for name, expected in {
        **manifest["source_sha256"],
        **manifest["environment_sha256"],
    }.items():
        if digest(root / name) != expected:
            raise ValueError(f"frozen source/environment changed: {name}")
    if (
        digest(root / manifest["candidate_path"] / "pilot.json")
        != manifest["candidate_sha256"]
    ):
        raise ValueError("candidate changed")
    return manifest


def seal(output, audit_path):
    output = Path(output)
    if Path(audit_path).resolve() != (output / "audit-review.json").resolve():
        raise ValueError("audit must be stored as audit-review.json in the package")
    if (output / "seal.json").exists():
        raise ValueError("refusing to overwrite a seal")
    verify_prepared(output)
    audit = json.loads(Path(audit_path).read_text())
    selected = json.loads((output / "audit-selection.json").read_text())
    if audit["selection_sha256"] != digest(output / "audit-selection.json"):
        raise ValueError("audit selection changed")
    reviews = {r["id"]: r for r in audit["reviews"]}
    if len(reviews) != len(audit["reviews"]) or set(reviews) != set(
        selected["sample_ids"]
    ):
        raise ValueError("audit must cover every selected pair exactly once")
    for key, review in reviews.items():
        if (
            review["row_sha256"] != selected["rows_sha256"][key]
            or review["verdict"] != "pass"
        ):
            raise ValueError("unresolved or stale audit review")
        if not review.get("notes") or not review.get("both_styles_inspected"):
            raise ValueError("audit review needs paired inspection notes")
    if not audit.get("reviewer") or audit.get("unresolved_findings"):
        raise ValueError("reviewer missing or unresolved findings")
    result = {
        "status": "frozen-prepared-not-launched",
        "manifest_sha256": digest(output / "manifest.json"),
        "audit_sha256": digest(audit_path),
        "reviewer": audit["reviewer"],
        "launch_gates": [
            "Verify seal, source, environment, dataset and tokenizer hashes on the GPU host.",
            "Run pinned Axolotl preprocess and inspect the actual assistant-only loss mask.",
            "Validate GPU model loading, QLoRA targets and an inference/structured-decoding smoke test.",
            "No sbatch or training has been launched by preparation.",
        ],
    }
    (output / "seal.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--candidate", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--tokenizer", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("output", type=Path)
    sealing = sub.add_parser("seal")
    sealing.add_argument("output", type=Path)
    sealing.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.candidate, args.output, args.tokenizer)
    elif args.command == "verify":
        verify_prepared(args.output)
        print("Prepared hashes verified.")
    else:
        print(json.dumps(seal(args.output, args.audit), indent=2))


if __name__ == "__main__":
    main()
