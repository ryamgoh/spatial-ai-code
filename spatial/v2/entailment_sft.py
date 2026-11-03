"""Explicit four-run SFT preparation and verification; never generates training data."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from spatial.v2.composition_evaluation import adapter_hash, parse_or_none
from spatial.v2.entailment_generation import GenerationWorkflow, digest, read_jsonl
from spatial.v2.entailment_release import verify_release
from spatial.v2.entailment_thinking_training import require, training_config
from spatial.v2.serialization import atomic_json

STYLES = ("natural", "symbolic")
SEEDS = (42, 43)
SCHEMA = "spatialentail-sft-campaign-v1"
AXOLOTL_REVISION = "7d775808d0ff71f21f553bd7d761631d7fb089e6"


def run_key(style, seed):
    require(
        style in STYLES and type(seed) is int and seed in SEEDS, "unknown style/seed"
    )
    return f"{style}-seed-{seed}"


def wandb_campaign_id(campaign):
    return "sft-" + hashlib.sha256(str(campaign.resolve()).encode()).hexdigest()[:16]


def config_for(model, artifacts, directory, style, seed):
    key = run_key(style, seed)
    campaign = wandb_campaign_id(directory.parent)
    config = training_config(model, style, str(directory))
    config.update(
        tokenizer_config=str(artifacts / "tokenizers/active"),
        output_dir=str(directory / "model"),
        datasets=[
            {
                "path": str(artifacts / f"data/train/{style}-tokenized.jsonl"),
                "ds_type": "json",
                "type": None,
                "split": "train",
            }
        ],
        test_datasets=[
            {
                "path": str(artifacts / f"data/dev/{style}-tokenized.jsonl"),
                "ds_type": "json",
                "type": None,
                "split": "train",
            }
        ],
        dataset_prepared_path=str(directory / "cache/prepared"),
        skip_prepare_dataset=True,
        dataset_exact_deduplication=False,
        remove_unused_columns=True,
        eval_strategy="epoch",
        eval_sample_packing=False,
        eval_batch_size=1,
        save_only_model=True,
        seed=seed,
        data_seed=seed,
        accelerator_config={"use_seedable_sampler": True},
        lora_dropout=0.0,
        lora_target_linear=False,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        bnb_config_kwargs={
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_storage": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        },
        plugins=["spatial.v2.entailment_sft_plugin.FinalSFTPlugin"],
        wandb_project="spatialentail-final-v1-sft",
        wandb_name=f"{campaign}-{key}",
        wandb_run_id=f"{campaign}-{key}",
    )
    return config


def prepare(generation_config, artifacts, output, *, release=None):
    artifacts, output = artifacts.resolve(), output.resolve()
    require(not output.exists() and output.parent.is_dir(), "use a fresh campaign path")
    workflow = GenerationWorkflow(generation_config)
    if release is None:
        manifest = workflow.verify(artifacts)
    else:
        release = release.resolve()
        manifest = verify_release(
            artifacts,
            release,
            root=generation_config.root,
            schema=generation_config.manifest_schema,
        )
    # Preview configurations are useful for smoke artifacts, but cannot be launched.
    model = manifest["model"]
    sources = {
        **workflow.fingerprints(),
        **{
            name: digest(generation_config.root / name)
            for name in ("pyproject.toml", "uv.lock", "finetune/finetune.py")
        },
    }
    output.mkdir()
    files = {}
    for style in STYLES:
        for seed in SEEDS:
            directory = output / run_key(style, seed)
            directory.mkdir()
            metadata = {
                "style": style,
                "seed": seed,
                "model": model,
                "artifacts": str(artifacts),
                "root": str(generation_config.root),
                "generation_schema": generation_config.manifest_schema,
                "generation_sources": manifest["source_sha256"],
                "generation_manifest_sha256": digest(artifacts / "manifest.json"),
                "release": str(release) if release is not None else None,
                "release_sha256": digest(release) if release is not None else None,
            }
            atomic_json(directory / "run.json", metadata)
            (directory / "config.yaml").write_text(
                yaml.safe_dump(
                    config_for(model, artifacts, directory, style, seed),
                    sort_keys=False,
                )
            )
            for name in ("run.json", "config.yaml"):
                path = directory / name
                files[path.relative_to(output).as_posix()] = digest(path)
    campaign = {
        "schema": SCHEMA,
        "status": "prepared-not-trained",
        "source_sha256": sources,
        "files": files,
        "generation_manifest_sha256": digest(artifacts / "manifest.json"),
    }
    atomic_json(output / "campaign.json", campaign)
    for style in STYLES:
        for seed in SEEDS:
            verify_run(output / run_key(style, seed), require_ready=False)
    return campaign


def verify_run(directory, *, require_ready=True, training_revision=None):
    directory = directory.resolve()
    campaign = json.loads((directory.parent / "campaign.json").read_text())
    require(campaign["schema"] == SCHEMA, "wrong SFT campaign")
    expected_files = {
        f"{run_key(style, seed)}/{name}"
        for style in STYLES
        for seed in SEEDS
        for name in ("run.json", "config.yaml")
    }
    require(set(campaign["files"]) == expected_files, "incomplete four-run campaign")
    for name, sha in campaign["files"].items():
        require(
            digest(directory.parent / name) == sha, f"campaign file changed: {name}"
        )
    meta = json.loads((directory / "run.json").read_text())
    require(
        directory.name == run_key(meta["style"], meta["seed"]), "mixed run identity"
    )
    root, artifacts = Path(meta["root"]), Path(meta["artifacts"])
    if training_revision is not None:
        from spatial.v2.entailment_release import original_sources

        original_sources(root, campaign, training_revision)
    else:
        for name, sha in campaign["source_sha256"].items():
            path = (root / name).resolve()
            require(path.is_relative_to(root.resolve()), "invalid source path")
            require(digest(path) == sha, f"training source changed: {name}")
    require(
        digest(artifacts / "manifest.json")
        == meta["generation_manifest_sha256"]
        == campaign["generation_manifest_sha256"],
        "data release changed",
    )
    if meta.get("release") is not None:
        receipt = Path(meta["release"])
        require(
            digest(receipt) == meta["release_sha256"], "data release receipt changed"
        )
        manifest = verify_release(
            artifacts, receipt, root=root, schema=meta["generation_schema"]
        )
    else:
        require(
            not require_ready, "candidate is not a training release: provide --release"
        )
        sources = {name: digest(root / name) for name in meta["generation_sources"]}
        manifest = GenerationWorkflow.verify_saved(
            artifacts, sources, meta["generation_schema"]
        )
    require(manifest["model"] == meta["model"], "model differs from data release")
    actual = yaml.safe_load((directory / "config.yaml").read_text())
    require(
        actual
        == config_for(meta["model"], artifacts, directory, meta["style"], meta["seed"]),
        "unexpected SFT configuration",
    )
    return meta


def verify_environment():
    from importlib.metadata import distribution, version

    source = json.loads(distribution("axolotl").read_text("direct_url.json") or "{}")
    require(
        source.get("vcs_info", {}).get("commit_id") == AXOLOTL_REVISION,
        "use the repository-pinned Axolotl revision",
    )
    expected = {
        "torch": "2.13.0",
        "transformers": "5.16.1",
        "tokenizers": "0.23.1",
        "jinja2": "3.1.6",
        "peft": "0.20.0",
        "datasets": "4.8.4",
        "accelerate": "1.13.0",
        "bitsandbytes": "0.50.2",
    }
    for name, wanted in expected.items():
        require(version(name).split("+")[0] == wanted, f"wrong runtime version: {name}")


def train_run(directory):
    meta = verify_run(directory)
    require(
        not (directory / "model").exists(), "training output exists; resume forbidden"
    )
    verify_environment()
    subprocess.run(
        [
            sys.executable,
            str(Path(meta["root"]) / "finetune/finetune.py"),
            str(directory / "config.yaml"),
        ],
        cwd=meta["root"],
        env={**os.environ, "WANDB_RUN_GROUP": wandb_campaign_id(directory.parent)},
        check=True,
    )
    selected_checkpoint(directory)


def selected_checkpoint(directory, *, training_revision=None):
    meta = (
        verify_run(directory)
        if training_revision is None
        else verify_run(directory, training_revision=training_revision)
    )
    result = json.loads((directory / "selection.json").read_text())
    require(
        result["style"] == meta["style"]
        and result["seed"] == meta["seed"]
        and result["generation_manifest_sha256"] == meta["generation_manifest_sha256"],
        "selection belongs to another run",
    )
    require(
        result["checkpoint_reload"] == "both-epochs-passed",
        "checkpoint reload not verified",
    )
    dev_path = Path(meta["artifacts"]) / "data/dev/references.jsonl"
    dev = read_jsonl(dev_path)
    epochs = []
    for epoch in (1, 2):
        path = directory / f"dev-epoch-{epoch}.json"
        require(
            digest(path) == result["epoch_receipts_sha256"][str(epoch)],
            "dev epoch receipt changed",
        )
        item = json.loads(path.read_text())
        require(
            item["epoch"] == epoch
            and item["step"] == epoch * 750
            and item["dev_sha256"] == digest(dev_path),
            "epoch provenance differs",
        )
        predictions = directory / f"dev-epoch-{epoch}.jsonl"
        require(
            digest(predictions) == item["predictions_sha256"], "Dev predictions changed"
        )
        outputs = read_jsonl(predictions)
        require(
            [r["id"] for r in outputs] == [r["id"] for r in dev],
            "Dev prediction IDs differ",
        )
        require(
            sum(
                parse_or_none(p["text"], r["contract"], 1) == r["answer"]
                for p, r in zip(outputs, dev, strict=True)
            )
            == item["correct"],
            "Dev accuracy differs from saved predictions",
        )
        epochs.append(item)
    best = choose_epoch(epochs)
    require(
        result["epoch"] == best["epoch"], "selected epoch disagrees with dev results"
    )
    checkpoint = directory / "model" / f"checkpoint-{best['step']}"
    require(
        adapter_hash(checkpoint) == best["adapter_sha256"] == result["adapter_sha256"],
        "selected adapter changed",
    )
    return meta, checkpoint, best["epoch"]


def choose_epoch(epochs):
    from fractions import Fraction

    require(
        len(epochs) == 2 and {e["epoch"] for e in epochs} == {1, 2},
        "both dev epochs required",
    )
    for item in epochs:
        require(
            type(item["correct"]) is int
            and type(item["total"]) is int
            and item["total"] == 1200
            and 0 <= item["correct"] <= item["total"],
            "incomplete dev score",
        )
    return max(epochs, key=lambda e: (Fraction(e["correct"], e["total"]), -e["epoch"]))


def main(generation_config):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--artifacts", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    prep.add_argument("--release", type=Path)
    for name in ("verify", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True, type=Path)
        if name == "verify":
            command.add_argument("--selected", action="store_true")
        if name == "evaluate":
            command.add_argument("--benchmark", choices=("sft-bench", "external-test"))
            command.add_argument("--untuned", action="store_true")
            command.add_argument("--shots", type=int, choices=(0, 1, 3), default=0)
            command.add_argument("--training-revision")
            command.add_argument("--batch-size", type=int, default=16)
    check = commands.add_parser("verify-evaluation")
    check.add_argument("--run", required=True, type=Path)
    check.add_argument("--training-revision")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(generation_config, args.artifacts, args.output, release=args.release)
        print(f"Prepared four configurations at {args.output}; no jobs submitted.")
    elif args.command == "verify":
        if args.selected:
            selected_checkpoint(args.run.resolve())
        else:
            verify_run(args.run)
        print("Training release and four-run campaign verified.")
    elif args.command == "train":
        train_run(args.run.resolve())
    elif args.command == "verify-evaluation":
        from spatial.v2.entailment_final_evaluation import verify_campaign

        verify_campaign(args.run.resolve(), training_revision=args.training_revision)
        print("Selections, evaluation policy and all prompt budgets verified.")
    else:
        from spatial.v2.entailment_final_evaluation import evaluate_job

        require(args.batch_size > 0, "batch size must be positive")
        evaluate_job(
            args.run.resolve(),
            untuned=args.untuned,
            shots=args.shots,
            benchmark=args.benchmark,
            training_revision=args.training_revision,
            batch_size=args.batch_size,
        )
