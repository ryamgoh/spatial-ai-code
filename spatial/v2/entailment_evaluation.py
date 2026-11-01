"""Evaluate saved Qwen3 thinking adapters; reuse existing two-stage scoring."""

import argparse
import json
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path

from spatial.v2.composition_evaluation import adapter_hash, score
from spatial.v2.composition_training import messages
from spatial.v2.entailment_scene_protocol import SHORT_VERSION, parse_row
from spatial.v2.entailment_thinking_training import (
    EXPERIMENT,
    MODEL,
    REVISION,
    digest,
    read_jsonl,
    require,
    write_json,
)
from spatial.v2.entailment_unique_study import validate_row


def checkpoints(prepared, style):
    model_dir = prepared / "models" / f"{style}-seed-42"
    epochs = {}
    for path in model_dir.glob("checkpoint-*"):
        state_path = path / "trainer_state.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text())
        epoch = state.get("epoch", 0)
        if epoch not in (1, 2):
            continue
        require(int(epoch) not in epochs, "multiple checkpoints for the same epoch")
        adapter_hash(path)
        config = json.loads((path / "adapter_config.json").read_text())
        require(config.get("base_model_name_or_path") == MODEL, "adapter base mismatch")
        epochs[int(epoch)] = path.resolve()
    require(set(epochs) == {1, 2}, f"two epoch checkpoints required: {model_dir}")
    return epochs


def discover(prepared=None):
    if prepared is not None:
        path = Path(prepared).resolve()
        for style in ("natural", "symbolic"):
            checkpoints(path, style)
        return path
    candidates = []
    for path in (EXPERIMENT / "runs").glob("thinking-*/prepared"):
        try:
            for style in ("natural", "symbolic"):
                checkpoints(path, style)
            candidates.append(path.resolve())
        except (ValueError, FileNotFoundError, KeyError):
            continue
    require(
        len(candidates) == 1,
        "Expected one complete training run; set PREPARED if several exist: "
        + ", ".join(map(str, candidates)),
    )
    return candidates[0]


def load_benchmark(directory, split):
    directory = Path(directory)
    require(
        (directory / "manifest.json").is_file(),
        f"Missing materialized benchmark: {directory}. "
        "The old scoped source is not an evaluation release.",
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    hashes = (
        manifest.get("payload_sha256")
        or manifest.get("output_hashes")
        or manifest.get("output_sha256", {})
    )
    require(
        "references.jsonl" in hashes, "benchmark references must have a manifest hash"
    )
    require(
        digest(directory / "references.jsonl") == hashes["references.jsonl"],
        "benchmark references changed",
    )
    rows = [
        r for r in read_jsonl(directory / "references.jsonl") if r["split"] == split
    ]
    expected = {
        "dev": 1200,
        "in-range-test": 2400,
        "extrapolation-test": 1200,
        "external-test": 1500,
    }[split]
    require(len({r["base_id"] for r in rows}) == expected, "wrong benchmark base count")
    require(len({r["id"] for r in rows}) == len(rows), "duplicate benchmark IDs")
    for row in rows:
        validate_row(row)
        require(
            row["prompt_version"] == SHORT_VERSION,
            "short system-rule protocol required",
        )
        parse_row(row)
    if split != "external-test":
        inventory = json.loads(
            (EXPERIMENT / "generation-exclusions-v1.json").read_text()
        )
        require(
            {r["base_id"] for r in rows} <= set(inventory["base_ids"]),
            "benchmark contains unselected bases",
        )
    return rows


def stage2_suffix(tokenizer, response_ids):
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    close = tokenizer.convert_tokens_to_ids("</think>")
    terminated = bool(response_ids) and response_ids[-1] == end
    # Raw replay preserves reasoning that the multi-turn template would remove.
    suffix = "" if close in response_ids or terminated else "\n</think>"
    suffix += "" if terminated else "<|im_end|>"
    return suffix + (
        "\n<|im_start|>user\nBased on the question and your reasoning above, "
        "give only the final answer.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def choose(runs):
    require(
        len(runs) == 2 and {r["epoch"] for r in runs} == {1, 2},
        "both dev epochs required",
    )
    for key in ("style", "benchmark_sha256", "model", "revision", "limits"):
        require(runs[0][key] == runs[1][key], f"dev provenance differs: {key}")
    require(
        all(r["split"] == "dev" and r["adapter"] for r in runs),
        "selection uses adapter dev results only",
    )
    return max(
        runs,
        key=lambda r: (
            Fraction(r["scores"]["stage1"]["correct"], r["scores"]["stage1"]["total"]),
            -r["epoch"],
        ),
    )


def run(
    engine, tokenizer, rows, directory, output, style, split, adapter, epoch, limits
):
    require(
        not output.exists() and not output.with_suffix(".partial.jsonl").exists(),
        f"existing results: {output}",
    )
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    fingerprint = adapter_hash(adapter) if adapter else None
    benchmark_hash = digest(directory / "references.jsonl")
    request = LoRARequest(style, epoch or 1, str(adapter)) if adapter else None
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".partial.jsonl").open("x") as journal:
        for row in rows:
            prefix = tokenizer.apply_chat_template(
                messages(row, style), tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer.encode(prefix, add_special_tokens=False)
            require(
                len(ids) + limits["stage1"] <= limits["context"], "Stage-1 overflow"
            )
            first = engine.generate(
                [{"prompt_token_ids": ids}],
                SamplingParams(
                    temperature=0,
                    max_tokens=limits["stage1"],
                    stop_token_ids=[tokenizer.eos_token_id],
                ),
                lora_request=request,
                use_tqdm=False,
            )[0].outputs[0]
            first_ids = list(first.token_ids)
            replay = (
                ids
                + first_ids
                + tokenizer.encode(
                    stage2_suffix(tokenizer, first_ids), add_special_tokens=False
                )
            )
            require(
                len(replay) + limits["stage2"] <= limits["context"],
                "Stage-2 overflow; no truncation",
            )
            second = engine.generate(
                [{"prompt_token_ids": replay}],
                SamplingParams(
                    temperature=0,
                    max_tokens=limits["stage2"],
                    structured_outputs=StructuredOutputsParams(choice=list("ABCDEF")),
                    stop_token_ids=[tokenizer.eos_token_id],
                ),
                lora_request=request,
                use_tqdm=False,
            )[0].outputs[0]
            raw = tokenizer.decode(first_ids, skip_special_tokens=False)
            final = raw.split("</think>", 1)[1] if "</think>" in raw else ""
            final = final.removesuffix("<|im_end|>").strip()
            record = {
                "id": row["id"],
                "stage1": {
                    "text": final,
                    "raw_text": raw,
                    "token_ids": first_ids,
                    "input_tokens": len(ids),
                    "finish_reason": first.finish_reason,
                },
                "stage2": {
                    "text": second.text,
                    "token_ids": list(second.token_ids),
                    "input_tokens": len(replay),
                    "finish_reason": second.finish_reason,
                },
            }
            records.append(record)
            journal.write(json.dumps(record) + "\n")
            journal.flush()
            if len(records) % 100 == 0:
                print(
                    f"{style}/{split}/epoch-{epoch}: {len(records)}/{len(rows)}",
                    flush=True,
                )
    result = {
        "model": MODEL,
        "revision": REVISION,
        "style": style,
        "split": split,
        "adapter": str(adapter) if adapter else None,
        "adapter_sha256": fingerprint,
        "epoch": epoch,
        "limits": limits,
        "benchmark_sha256": benchmark_hash,
        "outputs": records,
        "scores": score(rows, records),
        "runtime_versions": {n: version(n) for n in ("torch", "vllm", "transformers")},
    }
    require(not adapter or adapter_hash(adapter) == fingerprint, "adapter changed")
    require(
        digest(directory / "references.jsonl") == result["benchmark_sha256"],
        "benchmark changed",
    )
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dev", "selected", "untuned"))
    parser.add_argument("--style", required=True, choices=("natural", "symbolic"))
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("dev", "in-range-test", "extrapolation-test", "external-test"),
    )
    args = parser.parse_args()
    prepared = discover(args.prepared)
    split = "dev" if args.mode == "dev" else args.split
    require(
        split is not None and (args.mode == "dev" or split != "dev"),
        "specify a final test split",
    )
    rows = load_benchmark(args.benchmark, split)
    model = json.loads((prepared / "model-selection.json").read_text())
    require(
        model["model_id"] == MODEL and model["revision"] == REVISION,
        "training model differs",
    )
    limits = {
        "context": model["context_length"],
        "stage1": model["stage1_max_new_tokens"],
        "stage2": model["stage2_max_new_tokens"],
    }
    out = prepared.parent / "evaluation"
    selection_path = out / f"selected-{args.style}.json"
    if args.mode == "dev":
        require(not selection_path.exists(), "existing selection; use existing results")
        adapters = checkpoints(prepared, args.style)
    elif args.mode == "selected":
        require(selection_path.is_file(), "run dev selection first")
        selection = json.loads(selection_path.read_text())
        require(
            selection["style"] == args.style
            and selection["limits"] == limits
            and selection["model"] == MODEL
            and selection["revision"] == REVISION,
            "selection provenance mismatch",
        )
        path = Path(selection["adapter"])
        require(
            adapter_hash(path) == selection["adapter_sha256"],
            "selected adapter changed",
        )
        adapters = {selection["epoch"]: path}
    else:
        adapters = {None: None}
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    engine = LLM(
        model=MODEL,
        revision=REVISION,
        tokenizer_revision=REVISION,
        dtype="bfloat16",
        max_model_len=limits["context"],
        enable_lora=args.mode != "untuned",
        max_lora_rank=64,
        seed=42,
        trust_remote_code=False,
    )
    results = [
        run(
            engine,
            tokenizer,
            rows,
            args.benchmark,
            out / f"{args.style}-{args.mode}-{split}-epoch-{epoch}.json",
            args.style,
            split,
            adapter,
            epoch,
            limits,
        )
        for epoch, adapter in adapters.items()
    ]
    if args.mode == "dev":
        selected = choose(results)
        write_json(
            selection_path,
            {k: v for k, v in selected.items() if k not in ("outputs", "scores")},
        )
        print(f"Selected {args.style} epoch {selected['epoch']}: {selected['adapter']}")


if __name__ == "__main__":
    main()
