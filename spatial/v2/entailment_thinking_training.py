"""Prepare audited Unique-only training data for native Qwen3 thinking."""

import argparse
import hashlib
import json
import random
import re
import shutil
import tempfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/spatialentail-composition-v1"
VERSION = "unique-thinking-training-v1"
STYLES = ("natural", "symbolic")
TOKENIZER_VERSIONS = {
    "transformers": "5.16.1",
    "tokenizers": "0.23.1",
    "jinja2": "3.1.6",
}
MODEL = "Qwen/Qwen3-4B-Thinking-2507"
REVISION = "768f209d9ea81521153ed38c47d515654e938aea"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def write_jsonl(path, values):
    with Path(path).open("w") as stream:
        stream.writelines(json.dumps(value) + "\n" for value in values)


def validate_model(model):
    require(model["model_id"] == model["tokenizer_id"] == MODEL, "wrong model")
    require(
        model["revision"] == model["tokenizer_revision"] == REVISION,
        "wrong model/tokenizer revision",
    )
    require(model["thinking_mode"] == "thinking-only", "thinking mode required")
    require(model["chat_template_kwargs"] == {}, "do not override thinking mode")
    require(model["arms"] == list(STYLES), "matched arms required")
    require(
        model["training_candidate"] == "unique-training-candidate-v3", "wrong source"
    )
    for key in ("context_length", "stage1_max_new_tokens", "stage2_max_new_tokens"):
        require(type(model[key]) is int and model[key] > 0, f"invalid {key}")


def thinking_messages(source, answer):
    require(
        [m["role"] for m in source] == ["system", "user", "assistant"],
        "expected one system/user/assistant turn",
    )
    require(re.fullmatch("[A-F]", answer) is not None, "invalid gold answer")
    for message in source:
        require(
            all(
                tag not in message["content"]
                for tag in (
                    "<think>",
                    "</think>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|endoftext|>",
                )
            ),
            "source contains reserved template tokens",
        )
    target = source[-1]["content"]
    proof, separator, final = target.rpartition("\n\n")
    require(separator and proof.strip() == proof and proof, "missing explanation")
    require(final == f"Answer: {answer}", "reference answer mismatch")
    require("Answer:" not in proof, "multiple answer lines")
    return [
        *source[:-1],
        {"role": "assistant", "reasoning_content": proof, "content": final},
    ]


def audit_tokenized(tokenizer, record, prefix, suffix):
    ids, labels, attention = (
        record["input_ids"],
        record["labels"],
        record["attention_mask"],
    )
    require(len(ids) == len(labels) == len(attention), "unequal token array lengths")
    require(all(x == 1 for x in attention), "unexpected padding")
    boundary = next((i for i, x in enumerate(labels) if x != -100), len(labels))
    require(0 < boundary < len(ids), "missing prompt or supervised target")
    require(labels[:boundary] == [-100] * boundary, "prompt is supervised")
    require(
        labels[boundary:] == ids[boundary:], "assistant tokens are masked or altered"
    )
    decode = lambda tokens: tokenizer.decode(tokens, skip_special_tokens=False)
    require(decode(ids[:boundary]) == prefix, "decoded prompt boundary mismatch")
    require(decode(ids[boundary:]) == suffix, "decoded target boundary mismatch")
    require(
        labels.count(tokenizer.convert_tokens_to_ids("</think>")) == 1,
        "thinking terminator must be supervised exactly once",
    )
    require(
        labels.count(tokenizer.eos_token_id) == 1,
        "assistant turn terminator must be supervised exactly once",
    )
    return boundary


def tokenize(tokenizer, source, answer, model):
    messages = thinking_messages(source, answer)
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    require(prefix.endswith("<|im_start|>assistant\n<think>\n"), "wrong native prefix")
    suffix = (
        messages[-1]["reasoning_content"]
        + "\n</think>\n\n"
        + messages[-1]["content"]
        + "<|im_end|>\n"
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    require(full == prefix + suffix, "training and inference templates disagree")
    ids = tokenizer.encode(full, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    require(ids[: len(prompt_ids)] == prompt_ids, "token boundary crosses loss mask")
    row = {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prompt_ids) + ids[len(prompt_ids) :],
    }
    boundary = audit_tokenized(tokenizer, row, prefix, suffix)
    require(len(ids) <= model["context_length"], "training context overflow")
    require(
        len(ids) - boundary <= model["stage1_max_new_tokens"],
        "supervised target exceeds Stage-1 cap",
    )
    require(
        boundary + model["stage1_max_new_tokens"] <= model["context_length"],
        "prompt plus Stage-1 reserve exceeds context",
    )
    return (
        row,
        messages,
        {
            "prompt_tokens": boundary,
            "supervised_tokens": len(ids) - boundary,
            "training_tokens": len(ids),
            "stage1_reserved_tokens": boundary + model["stage1_max_new_tokens"],
        },
    )


def training_config(model, style, prefix):
    require(style in STYLES, "unknown style")
    run_key = hashlib.sha256(prefix.encode()).hexdigest()[:12]
    return {
        "base_model": model["model_id"],
        "revision_of_model": model["revision"],
        "tokenizer_config": f"{prefix}/tokenizer",
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
        "sequence_len": model["context_length"],
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
        "logging_nan_inf_filter": False,
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
        "use_wandb": True,
        "wandb_project": "spatialentail-composition-v1-sft",
        "wandb_mode": "online",
        "wandb_name": f"qwen3-4b-thinking-{style}-seed-42-{run_key}",
        "wandb_run_id": f"composition-{run_key}-{style}-42",
    }


def matched_configs(configs):
    normalized = []
    for style in STYLES:
        config = dict(configs[style])
        for key in (
            "output_dir", "datasets", "dataset_prepared_path",
            "wandb_name", "wandb_run_id",
        ):
            config.pop(key)
        normalized.append(config)
    require(normalized[0] == normalized[1], "training arms have different settings")


def load_source(candidate):
    manifest = json.loads((candidate / "manifest.json").read_text())
    require(manifest["examples_per_style"] == manifest["bases"] == 6000, "wrong size")
    require(
        manifest["audit_summary"]["status"] == "references-rerendered-and-audited"
        and manifest["audit_summary"]["statuses"] == {"passed": 6000}
        and not manifest["audit_summary"]["failures"],
        "source audit did not pass",
    )
    for name, sha in manifest["payload_sha256"].items():
        require(digest(candidate / name) == sha, f"source payload changed: {name}")
    refs = read_jsonl(candidate / "references.jsonl")
    require(
        len(refs) == len({r["id"] for r in refs}) == 6000, "duplicate/missing views"
    )
    require(len({r["base_id"] for r in refs}) == 6000, "duplicate base question")
    require(
        all(r["split"] == "train" and r["contract"] == "unique-entailed" for r in refs),
        "nontraining or non-Unique row",
    )
    counts = Counter(
        (r["query_kind"], "A-D" if r["answer"] in "ABCD" else r["answer"]) for r in refs
    )
    require(
        counts
        == Counter(
            {
                (kind, category): count
                for kind in ("direction", "which", "count")
                for category, count in (("A-D", 1200), ("E", 600), ("F", 200))
            }
        ),
        "training answer allocation changed",
    )
    source = {}
    for style in STYLES:
        rows = read_jsonl(candidate / f"{style}-training.jsonl")
        require(len(rows) == 6000, "wrong chat row count")
        require(
            [r["id"] for r in rows] == [r["id"] for r in refs], "source order mismatch"
        )
        for row, ref in zip(rows, refs, strict=True):
            require(row["base_id"] == ref["base_id"], "base mismatch")
            require(row["messages"][1]["content"] == ref["prompt"], "prompt mismatch")
            require(
                row["messages"][2]["content"] == ref["references"][style],
                "reference mismatch",
            )
        source[style] = {r["id"]: r for r in rows}
    return refs, source, manifest


def prepare(active_path, output, *, candidate=None):
    require(not output.exists(), "refusing to overwrite prepared output")
    model = json.loads(active_path.read_text())
    validate_model(model)
    require(
        all(version(name) == expected for name, expected in TOKENIZER_VERSIONS.items()),
        "use the pinned Transformers/tokenizers/Jinja2 versions",
    )
    candidate = (
        EXPERIMENT / model["training_candidate"]
        if candidate is None
        else Path(candidate).resolve()
    )
    refs, source, source_manifest = load_source(candidate)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = Path(
        snapshot_download(
            model["tokenizer_id"],
            revision=model["tokenizer_revision"],
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "config.json",
                "chat_template.jinja",
                "special_tokens_map.json",
            ],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    require(tokenizer.eos_token == "<|im_end|>", "wrong EOS token")
    refs.sort(key=lambda row: row["id"])
    random.Random(42).shuffle(refs)
    prefix = output.resolve().relative_to(ROOT).as_posix()
    configs = {style: training_config(model, style, prefix) for style in STYLES}
    matched_configs(configs)
    protected_paths = [
        candidate / "manifest.json",
        *(candidate / name for name in source_manifest["payload_sha256"]),
        EXPERIMENT / "unique-only-selection-v1/selection.json",
    ]
    generation_manifest = candidate / "generation.json"
    if generation_manifest.exists():
        protected_paths.append(generation_manifest)
        generation = json.loads(generation_manifest.read_text())
        for name, expected in generation["inputs_sha256"].items():
            require(
                digest(ROOT / name) == expected, f"generation input changed: {name}"
            )
            protected_paths.append(ROOT / name)
    protected = {str(p.relative_to(ROOT)): digest(p) for p in protected_paths}
    with tempfile.TemporaryDirectory(
        prefix="spatial-thinking-preparation-"
    ) as temporary:
        stage = Path(temporary)
        (stage / "data").mkdir()
        (stage / "configs").mkdir()
        (stage / "tokenizer").mkdir()
        for path in snapshot.iterdir():
            if path.is_file() and path.suffix in {".json", ".txt", ".jinja"}:
                shutil.copyfile(path, stage / "tokenizer" / path.name)
        measurements, previews = [], []
        for style in STYLES:
            token_rows, chat_rows = [], []
            for index, ref in enumerate(refs):
                original = source[style][ref["id"]]
                row, messages, measured = tokenize(
                    tokenizer, original["messages"], ref["answer"], model
                )
                token_rows.append(row)
                chat_rows.append(
                    {"id": ref["id"], "base_id": ref["base_id"], "messages": messages}
                )
                measurements.append({"id": ref["id"], "style": style, **measured})
                if index == 0:
                    previews.append(
                        {
                            "style": style,
                            "id": ref["id"],
                            **measured,
                            "masked_prompt": tokenizer.decode(
                                row["input_ids"][: measured["prompt_tokens"]]
                            ),
                            "supervised_text": tokenizer.decode(
                                row["input_ids"][measured["prompt_tokens"] :]
                            ),
                        }
                    )
            write_jsonl(stage / "data" / f"{style}-train-tokenized.jsonl", token_rows)
            write_jsonl(stage / "data" / f"{style}-train-messages.jsonl", chat_rows)
            require(
                read_jsonl(stage / "data" / f"{style}-train-tokenized.jsonl")
                == token_rows,
                "tokenized export read-back mismatch",
            )
            (stage / "configs" / f"{style}.yaml").write_text(
                yaml.safe_dump(configs[style], sort_keys=False)
            )
            print(
                f"{style}: 6000 native-template and loss-mask checks passed", flush=True
            )
        write_json(stage / "model-selection.json", model)
        write_jsonl(
            stage / "data/order.jsonl",
            [{"id": r["id"], "base_id": r["base_id"]} for r in refs],
        )
        write_jsonl(stage / "data/token-measurements.jsonl", measurements)
        write_json(stage / "mask-examples.json", previews)
        summary = {
            style: {
                key: {
                    "max": max(r[key] for r in measurements if r["style"] == style),
                    "total": sum(r[key] for r in measurements if r["style"] == style),
                }
                for key in (
                    "prompt_tokens",
                    "supervised_tokens",
                    "training_tokens",
                    "stage1_reserved_tokens",
                )
            }
            for style in STYLES
        }
        write_json(stage / "token-summary.json", summary)
        require(
            all(digest(ROOT / p) == sha for p, sha in protected.items()),
            "protected inputs changed during preparation",
        )
        manifest = {
            "schema": VERSION,
            "status": "locally-validated-not-launched",
            "model": model["model_id"],
            "revision": model["revision"],
            "examples_per_style": 6000,
            "distinct_questions": 6000,
            "versions": {name: version(name) for name in TOKENIZER_VERSIONS},
            "training_started": False,
            "sealed": False,
            "protected_sha256": protected,
            "source_sha256": {
                name: digest(ROOT / name)
                for name in (
                    "spatial/v2/entailment_thinking_training.py",
                    "finetune/finetune.py",
                    "pyproject.toml",
                    "uv.lock",
                )
            },
            "training_method": "QLoRA from standard BF16 checkpoint; 4-bit loading, BF16 compute",
            "loss_policy": "mask native generation prefix; supervise reasoning, closing think, answer and turn end",
            "selection": "Unique Stage-1 synthetic dev view accuracy; earliest epoch wins ties",
            "development_evaluation_status": "fixed selection preserved; current-model evaluation package pending",
            "stage2_status": "not validated by training preparation; native-thinking replay integration pending",
            "gpu_validation": "opening steps of actual training; no separate smoke-test job",
            "actual_axolotl_loader_validated": False,
            "baseline_generation_budget_validated": False,
            "files": {
                p.relative_to(stage).as_posix(): digest(p)
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            },
        }
        write_json(stage / "manifest.json", manifest)
        shutil.copytree(stage, output)
    verify(output)
    return manifest


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text())
    require(manifest["schema"] == VERSION, "unknown prepared package")
    for name, sha in manifest["files"].items():
        require(digest(output / name) == sha, f"prepared file changed: {name}")
    for section in ("protected_sha256", "source_sha256"):
        for name, sha in manifest[section].items():
            require(digest(ROOT / name) == sha, f"input/source changed: {name}")
    model = json.loads((output / "model-selection.json").read_text())
    validate_model(model)
    active = json.loads((EXPERIMENT / "active-model.json").read_text())
    validate_model(active)
    for key in (
        "model_id",
        "revision",
        "tokenizer_id",
        "tokenizer_revision",
        "context_length",
        "stage1_max_new_tokens",
        "stage2_max_new_tokens",
    ):
        require(active[key] == model[key], f"active setting changed: {key}")
    configs = {
        style: yaml.safe_load((output / "configs" / f"{style}.yaml").read_text())
        for style in STYLES
    }
    matched_configs(configs)
    prefix = output.resolve().relative_to(ROOT).as_posix()
    require(
        configs == {s: training_config(model, s, prefix) for s in STYLES},
        "unexpected trainer configuration",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--active", type=Path, default=EXPERIMENT / "active-model.json")
    parser.add_argument(
        "--output", type=Path, default=EXPERIMENT / "prepared-thinking-v1"
    )
    args = parser.parse_args()
    result = (
        prepare(args.active, args.output)
        if args.command == "prepare"
        else verify(args.output)
    )
    print(json.dumps({"status": result["status"], "examples_per_style": 6000}))


if __name__ == "__main__":
    main()
