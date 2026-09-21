"""Measure pre-GRPO reward variance with four stochastic rollouts per prompt."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ANSWER_RE = re.compile(r"Answer:\s*([A-E](?:\s*,\s*[A-E])*)", re.IGNORECASE)


def letters(value: object) -> set[str]:
    return {
        token.strip().upper()
        for token in re.split(r"[,;| ]+", str(value or ""))
        if token.strip().upper() in {"A", "B", "C", "D", "E"}
    }


def predicted(text: str) -> set[str]:
    matches = ANSWER_RE.findall(text)
    return letters(matches[-1]) if matches else set()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-prompts", type=int, default=48)
    parser.add_argument("--n-generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=14002)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_jsonl(args.data)
    rng = random.Random(args.seed)
    selected = rng.sample(rows, min(args.n_prompts, len(rows)))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        for row in selected
    ]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
    )
    params = SamplingParams(
        n=args.n_generations,
        temperature=0.7,
        top_p=0.95,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, params)

    details = []
    total_correct = total_rollouts = mixed_groups = all_correct = all_wrong = 0
    cell_totals: Counter = Counter()
    cell_correct: Counter = Counter()
    for row, output in zip(selected, outputs, strict=True):
        gold = letters(row["oracle_option"])
        completions = [candidate.text for candidate in output.outputs]
        correctness = [predicted(text) == gold for text in completions]
        correct = sum(correctness)
        total_correct += correct
        total_rollouts += len(correctness)
        mixed_groups += int(0 < correct < len(correctness))
        all_correct += int(correct == len(correctness))
        all_wrong += int(correct == 0)
        cell = str(row.get("generation_cell") or "unknown")
        cell_totals[cell] += len(correctness)
        cell_correct[cell] += correct
        details.append(
            {
                "generation_cell": cell,
                "oracle_option": row["oracle_option"],
                "correct_rollouts": correct,
                "num_rollouts": len(correctness),
                "predictions": [sorted(predicted(text)) for text in completions],
                "completions": completions,
            }
        )

    groups = len(selected)
    pass_rate = total_correct / total_rollouts if total_rollouts else 0.0
    mixed_rate = mixed_groups / groups if groups else 0.0
    checks = {
        "rollout pass rate is between 15% and 90%": 0.15 <= pass_rate <= 0.90,
        "at least 15% of prompt groups have mixed rewards": mixed_rate >= 0.15,
        "at least 4 prompt groups are not all correct": groups - all_correct >= 4,
    }
    report = {
        "model": args.model,
        "num_prompts": groups,
        "generations_per_prompt": args.n_generations,
        "pass_rate": pass_rate,
        "mixed_group_rate": mixed_rate,
        "mixed_groups": mixed_groups,
        "all_correct_groups": all_correct,
        "all_wrong_groups": all_wrong,
        "checks": checks,
        "go": all(checks.values()),
        "cell_accuracy": {
            key: cell_correct[key] / cell_totals[key] for key in sorted(cell_totals)
        },
        "details": details,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "details"}, indent=2
        )
    )
    if not report["go"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
