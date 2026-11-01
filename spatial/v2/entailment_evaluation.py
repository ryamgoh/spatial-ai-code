"""Evaluate saved Qwen3 thinking adapters; reuse existing two-stage scoring."""

import argparse
import fcntl
import json
import time
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
        runs[0].get("execution") == runs[1].get("execution"),
        "dev execution settings differ",
    )
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


def print_summary(result):
    label = f"{result['style']}/{result['split']}/epoch-{result['epoch']}"
    for stage in ("stage1", "stage2"):
        scores = result["scores"][stage]
        tokens = sum(len(r[stage]["token_ids"]) for r in result["outputs"])
        accuracy = scores["correct"] / scores["total"]
        print(
            f"{label} {stage}: {scores['correct']}/{scores['total']} correct "
            f"({accuracy:.2%}); generated tokens={tokens}; "
            f"mean={tokens / scores['total']:.1f}",
            flush=True,
        )


def run(
    engine,
    tokenizer,
    rows,
    directory,
    output,
    style,
    split,
    adapter,
    epoch,
    limits,
    batch_size=16,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    # A second submission must not append to the same result journal.
    with output.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"evaluation already running: {output}") from error
        return run_locked(
            engine,
            tokenizer,
            rows,
            directory,
            output,
            style,
            split,
            adapter,
            epoch,
            limits,
            batch_size,
        )


def run_locked(
    engine,
    tokenizer,
    rows,
    directory,
    output,
    style,
    split,
    adapter,
    epoch,
    limits,
    batch_size=16,
):
    require(batch_size > 0, "batch size must be positive")
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    fingerprint = adapter_hash(adapter) if adapter else None
    benchmark_hash = digest(directory / "references.jsonl")
    provenance = {
        "model": MODEL,
        "revision": REVISION,
        "style": style,
        "split": split,
        "adapter": str(adapter) if adapter else None,
        "adapter_sha256": fingerprint,
        "epoch": epoch,
        "limits": limits,
        "benchmark_sha256": benchmark_hash,
    }
    if output.exists():
        result = json.loads(output.read_text())
        require(
            all(result.get(k) == v for k, v in provenance.items()),
            "completed result provenance mismatch",
        )
        require(
            [r["id"] for r in result["outputs"]] == [r["id"] for r in rows],
            "completed result IDs differ",
        )
        require(
            result["scores"] == score(rows, result["outputs"]),
            "completed result scores differ",
        )
        print(f"Already completed: {output}", flush=True)
        print_summary(result)
        return result
    execution = {
        "runner": "batched-token-replay-v1",
        "batch_size": batch_size,
        "temperature": 0,
        "seed": 42,
        "runtime_versions": {n: version(n) for n in ("torch", "vllm", "transformers")},
    }
    metadata = {**provenance, "execution": execution}
    partial = output.with_suffix(".partial.jsonl")
    receipt = output.with_suffix(".resume.json")
    if receipt.exists():
        require(
            json.loads(receipt.read_text()) == metadata,
            "resume provenance/execution mismatch; do not mix runs",
        )
    else:
        require(
            not partial.exists(),
            f"Legacy partial result cannot be resumed safely; move aside: {partial}",
        )
        write_json(receipt, metadata)
    request = LoRARequest(style, epoch or 1, str(adapter)) if adapter else None
    records = []
    if partial.exists():
        # SIGTERM can interrupt the final write. Only drop an unterminated tail.
        with partial.open("rb+") as journal:
            while line := journal.readline():
                if not line.endswith(b"\n"):
                    journal.truncate(journal.tell() - len(line))
                    break
                records.append(json.loads(line))
        require(
            [r["id"] for r in records] == [r["id"] for r in rows[: len(records)]],
            "partial result IDs/order differ",
        )
    label = f"{style}/{split}/{'untuned' if epoch is None else f'epoch-{epoch}'}"
    print(
        f"{label}: {len(records)}/{len(rows)} completed; batch_size={batch_size}",
        flush=True,
    )
    started = time.monotonic()
    initial_count = len(records)
    with partial.open("a") as journal:
        for offset in range(len(records), len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            prompts = []
            for row in batch:
                prefix = tokenizer.apply_chat_template(
                    messages(row, style), tokenize=False, add_generation_prompt=True
                )
                ids = tokenizer.encode(prefix, add_special_tokens=False)
                require(
                    len(ids) + limits["stage1"] <= limits["context"], "Stage-1 overflow"
                )
                prompts.append(ids)
            print(
                f"{label}: Stage 1, examples {offset + 1}-{offset + len(batch)}/{len(rows)}",
                flush=True,
            )
            first_outputs = engine.generate(
                [{"prompt_token_ids": ids} for ids in prompts],
                SamplingParams(
                    temperature=0,
                    max_tokens=limits["stage1"],
                    stop_token_ids=[tokenizer.eos_token_id],
                ),
                lora_request=request,
                use_tqdm=True,
            )
            require(len(first_outputs) == len(batch), "Stage-1 output count mismatch")
            firsts = [o.outputs[0] for o in first_outputs]
            replays = []
            for ids, first in zip(prompts, firsts, strict=True):
                first_ids = list(first.token_ids)
                replay = (
                    ids
                    + first_ids
                    + tokenizer.encode(
                        stage2_suffix(tokenizer, first_ids),
                        add_special_tokens=False,
                    )
                )
                require(
                    len(replay) + limits["stage2"] <= limits["context"],
                    "Stage-2 overflow; no truncation",
                )
                replays.append(replay)
            print(
                f"{label}: Stage 2, examples {offset + 1}-{offset + len(batch)}/{len(rows)}",
                flush=True,
            )
            second_outputs = engine.generate(
                [{"prompt_token_ids": replay} for replay in replays],
                SamplingParams(
                    temperature=0,
                    max_tokens=limits["stage2"],
                    structured_outputs=StructuredOutputsParams(choice=list("ABCDEF")),
                    stop_token_ids=[tokenizer.eos_token_id],
                ),
                lora_request=request,
                use_tqdm=True,
            )
            require(len(second_outputs) == len(batch), "Stage-2 output count mismatch")
            for row, ids, first, replay, second_output in zip(
                batch,
                prompts,
                firsts,
                replays,
                second_outputs,
                strict=True,
            ):
                second = second_output.outputs[0]
                first_ids = list(first.token_ids)
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
            elapsed = time.monotonic() - started
            rate = (len(records) - initial_count) / max(elapsed, 0.001)
            print(
                f"{label}: {len(records)}/{len(rows)} completed; "
                f"{rate:.2f} examples/s; ETA {(len(rows) - len(records)) / rate / 60:.1f} min",
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
        "execution": execution,
        "resumed_from_examples": initial_count,
    }
    require(not adapter or adapter_hash(adapter) == fingerprint, "adapter changed")
    require(
        digest(directory / "references.jsonl") == result["benchmark_sha256"],
        "benchmark changed",
    )
    write_json(output, result)
    print_summary(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dev", "selected", "untuned"))
    parser.add_argument("--style", required=True, choices=("natural", "symbolic"))
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--split",
        choices=("dev", "in-range-test", "extrapolation-test", "external-test"),
    )
    args = parser.parse_args()
    require(args.batch_size > 0, "batch size must be positive")
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
            batch_size=args.batch_size,
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
