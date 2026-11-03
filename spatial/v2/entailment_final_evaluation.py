"""Final-v1 model-reusing evaluation with fixed, Train-only demonstrations."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from spatial.v2.composition_evaluation import score
from spatial.v2.composition_training import messages
from spatial.v2.entailment_evaluation import run, stage2_suffix
from spatial.v2.entailment_generation import digest, read_jsonl
from spatial.v2.entailment_release import COUNTS
from spatial.v2.entailment_sft import (
    SEEDS,
    STYLES,
    run_key,
    selected_checkpoint,
    verify_environment,
    verify_run,
)
from spatial.v2.entailment_thinking_training import require, thinking_messages
from spatial.v2.serialization import atomic_json

POLICY = {
    "version": "final-v1-evaluation-8192-fewshot-v1",
    "context": 16384,
    "stage1": 8192,
    "stage2": 16,
    "demonstrations": "train-hash-ranked-direction-which-count-v1",
}
BENCHMARKS = ("sft-bench", "external-test")


def demonstrations(artifacts):
    rows = read_jsonl(artifacts / "data/train/references.jsonl")
    require(len(rows) == COUNTS["train"], "incomplete demonstration source")
    require(all(r["split"] == "train" for r in rows), "Train-only demonstrations")
    require(len({r["id"] for r in rows}) == len(rows), "duplicate training IDs")
    chosen = []
    for kind in ("direction", "which", "count"):
        candidates = [
            r for r in rows if r["query_kind"] == kind and r["answer"] in "ABCD"
        ]
        require(candidates, f"missing demonstration query: {kind}")
        chosen.append(
            min(
                candidates,
                key=lambda r: hashlib.sha256(
                    f"{POLICY['demonstrations']}/{r['id']}".encode()
                ).hexdigest(),
            )
        )
    return chosen


def prompt_ids(tokenizer, row, style, demos):
    prefix = tokenizer.apply_chat_template(
        messages(row, style), tokenize=False, add_generation_prompt=True
    )
    if demos:
        system, marker, query = prefix.partition("<|im_start|>user\n")
        require(marker, "native user boundary missing")
        turns = []
        for demo in demos:
            source = [
                *messages(demo, style),
                {"role": "assistant", "content": demo["references"][style]},
            ]
            # Render each demonstration separately so Qwen retains its thinking.
            full = tokenizer.apply_chat_template(
                thinking_messages(source, demo["answer"]),
                tokenize=False,
                add_generation_prompt=False,
            )
            _, boundary, turn = full.partition("<|im_start|>user\n")
            require(boundary and "</think>" in turn, "demonstration thinking missing")
            turns.append(boundary + turn)
        prefix = system + "".join(turns) + marker + query
    return tokenizer.encode(prefix, add_special_tokens=False)


def tasks(directory, *, untuned=False, shots=0, benchmark=None):
    require(type(shots) is int and shots in (0, 1, 3), "shots must be 0, 1 or 3")
    require(untuned or shots == 0, "tuned evaluation is zero-shot")
    require(benchmark is None or benchmark in BENCHMARKS, "unknown benchmark")
    datasets = (benchmark,) if benchmark else BENCHMARKS
    styles = STYLES if untuned else (directory.name.rsplit("-seed-", 1)[0],)
    require(all(style in STYLES for style in styles), "unknown evaluation style")
    return [(style, name) for style in styles for name in datasets]


def prepare_tasks(
    directory, tokenizer, meta, *, untuned=False, shots=0, benchmark=None
):
    artifacts = Path(meta["artifacts"])
    demos = demonstrations(artifacts)[:shots] if shots else []
    reserve = max(
        len(
            tokenizer.encode(
                stage2_suffix(tokenizer, response), add_special_tokens=False
            )
        )
        for response in ([], [tokenizer.convert_tokens_to_ids("</think>")])
    )
    prepared = []
    for style, name in tasks(
        directory, untuned=untuned, shots=shots, benchmark=benchmark
    ):
        data = artifacts / "data" / name
        rows = read_jsonl(data / "references.jsonl")
        require(len(rows) == COUNTS[name], "incomplete benchmark")
        require(
            not ({r["id"] for r in rows} & {r["id"] for r in demos}), "demo leakage"
        )
        lengths = [len(prompt_ids(tokenizer, r, style, demos)) for r in rows]
        require(
            max(lengths) + POLICY["stage1"] + reserve + POLICY["stage2"]
            <= POLICY["context"],
            f"{style}/{name}/{shots}-shot: context overflow; no truncation",
        )
        prepared.append((style, name, data, rows))
    return demos, prepared


def verify_campaign(campaign, *, training_revision=None):
    from transformers import AutoTokenizer

    metas = {}
    for seed in SEEDS:
        for style in STYLES:
            meta, _, _ = selected_checkpoint(
                campaign / run_key(style, seed), training_revision=training_revision
            )
            metas[style, seed] = meta
    meta = metas["natural", 42]
    tokenizer = AutoTokenizer.from_pretrained(
        Path(meta["artifacts"]) / "tokenizers/active", local_files_only=True
    )
    for seed in SEEDS:
        for style in STYLES:
            prepare_tasks(
                campaign / run_key(style, seed), tokenizer, metas[style, seed]
            )
    for shots in (0, 1, 3):
        prepare_tasks(
            campaign / run_key("natural", 42),
            tokenizer,
            meta,
            untuned=True,
            shots=shots,
        )


def token_profile(outputs, tokenizer):
    close = tokenizer.convert_tokens_to_ids("</think>")
    lengths = {
        stage: [len(r[stage]["token_ids"]) for r in outputs]
        for stage in ("stage1", "stage2")
    }
    lengths["thinking"] = [
        r["stage1"]["token_ids"].index(close)
        if close in r["stage1"]["token_ids"]
        else len(r["stage1"]["token_ids"])
        for r in outputs
    ]
    profile = {
        stage: {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": max(values),
        }
        for stage, values in lengths.items()
    }
    for stage in ("stage1", "stage2"):
        profile[stage]["length_finish_count"] = sum(
            r[stage]["finish_reason"] == "length" for r in outputs
        )
    profile["thinking"]["unclosed_count"] = sum(
        close not in r["stage1"]["token_ids"] for r in outputs
    )
    return profile


def evaluate_job(
    directory,
    *,
    untuned=False,
    shots=0,
    benchmark=None,
    training_revision=None,
    batch_size=16,
):
    tasks(directory, untuned=untuned, shots=shots, benchmark=benchmark)
    if untuned:
        meta = verify_run(directory, training_revision=training_revision)
        checkpoint, epoch = None, None
    else:
        meta, checkpoint, epoch = selected_checkpoint(
            directory, training_revision=training_revision
        )
    verify_environment()
    require(version("vllm") == "0.28.0", "use pinned vLLM")
    from transformers import AutoTokenizer
    from vllm import LLM

    artifacts, model = Path(meta["artifacts"]), meta["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        artifacts / "tokenizers/active", local_files_only=True
    )
    demos, prepared = prepare_tasks(
        directory, tokenizer, meta, untuned=untuned, shots=shots, benchmark=benchmark
    )
    root = Path(__file__).resolve().parents[2]
    campaign = json.loads((directory.parent / "campaign.json").read_text())
    sources = set(campaign["source_sha256"]) | {
        "spatial/v2/entailment_final_evaluation.py",
        "spatial/v2/entailment_evaluation.py",
        "experiments/spatialentail-final-v1/slurm/evaluate.sh",
        "experiments/spatialentail-final-v1/slurm/submit-evaluation.sh",
    }
    provenance = {
        "policy": POLICY,
        "shots": shots,
        "demonstration_ids": [r["id"] for r in demos],
        "train_references_sha256": digest(artifacts / "data/train/references.jsonl"),
        "training_revision": training_revision,
        "training_campaign_sha256": digest(directory.parent / "campaign.json"),
        "evaluation_source_sha256": {
            name: digest(root / name) for name in sorted(sources)
        },
    }
    # One engine per GPU job, reused for every benchmark/style in that job.
    engine = LLM(
        model=model["model_id"],
        revision=model["revision"],
        tokenizer=str(artifacts / "tokenizers/active"),
        dtype="bfloat16",
        max_model_len=POLICY["context"],
        enable_lora=not untuned,
        max_lora_rank=64,
        seed=42,
        gpu_memory_utilization=0.85,
    )
    limits = {key: POLICY[key] for key in ("context", "stage1", "stage2")}
    output_root = (
        directory.parent
        / "evaluation"
        / POLICY["version"]
        / (f"untuned-{shots}-shot" if untuned else directory.name)
    )
    for style, name, data, rows in prepared:
        output = output_root / f"{style}-{name}.json"
        result = run(
            engine,
            tokenizer,
            rows,
            data,
            output,
            style,
            name,
            checkpoint,
            epoch,
            limits,
            batch_size=batch_size,
            prompt_builder=lambda r, s=style: prompt_ids(tokenizer, r, s, demos),
            prompt_provenance=provenance,
        )
        breakdowns = {}
        for subset in sorted({r["split"] for r in rows}):
            selected = [r for r in rows if r["split"] == subset]
            ids = {r["id"] for r in selected}
            breakdowns[subset] = score(
                selected, [r for r in result["outputs"] if r["id"] in ids]
            )
        atomic_json(
            output.with_suffix(".summary.json"),
            {
                "style": style,
                "training_seed": None if untuned else meta["seed"],
                "evaluation": provenance,
                "limits": limits,
                "generation_manifest_sha256": meta["generation_manifest_sha256"],
                "result_sha256": digest(output),
                "scores": score(
                    rows,
                    result["outputs"],
                    extra_slice_fields=(
                        "split",
                        "construction_level",
                        "verified_depth",
                        "case",
                    ),
                ),
                "by_subgroup": breakdowns,
                "token_profile": token_profile(result["outputs"], tokenizer),
            },
        )
    require(
        all(
            digest(root / name) == sha
            for name, sha in provenance["evaluation_source_sha256"].items()
        ),
        "evaluation source changed during inference",
    )
