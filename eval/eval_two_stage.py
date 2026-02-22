"""
Two-stage evaluation for thinking models using vLLM guided decoding.

Stage 1: Free thinking
Stage 2: Constrained answer (A/B/C/D)

Usage:
    cd eval && uv run python eval_two_stage.py --config ../configs/models/eval_thinking_two_stage.yaml
"""

import argparse
import json
import re
from pathlib import Path

import yaml
from datasets import load_dataset
from vllm import LLM, SamplingParams

from utils import generate_datetime_id

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_spatial_eval():
    ds = load_dataset("MilaWang/SpatialEval", "tqa", split="test")
    ds = ds.filter(lambda x: bool(re.match(r"^spatialmap\.", x["id"])))
    for sample in ds:
        yield {
            "text": sample["text"],
            "target": sample["oracle_option"],
        }


def run_evaluation(config_path: Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    args = config["model_args"]
    llm = LLM(
        model=args["pretrained"],
        dtype=args.get("dtype", "auto"),
        gpu_memory_utilization=args.get("gpu_memory_utilization", 0.9),
        max_model_len=args.get("max_model_len", 4096),
        guided_decoding_backend="xgrammar",
    )

    thinking_max = config.get("thinking_max_tokens", 512)
    answer_max = config.get("answer_max_tokens", 2)

    output_dir = RESULTS_DIR / generate_datetime_id()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = list(load_spatial_eval())
    correct = 0

    for i, sample in enumerate(samples):
        prompt = f"{sample['text']}\n\nThink step by step."

        thinking = (
            llm.generate(
                [prompt], SamplingParams(max_tokens=thinking_max, temperature=0.0)
            )[0]
            .outputs[0]
            .text
        )

        answer_prompt = f"{prompt}\n{thinking}\n\nAnswer:"
        answer = (
            llm.generate(
                [answer_prompt],
                SamplingParams(
                    max_tokens=answer_max,
                    temperature=0.0,
                    guided_decoding={"choice": ["A", "B", "C", "D"]},
                ),
            )[0]
            .outputs[0]
            .text.strip()
        )

        is_correct = answer == sample["target"]
        if is_correct:
            correct += 1

        with open(output_dir / "samples.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "target": sample["target"],
                        "predicted": answer,
                        "correct": is_correct,
                        "thinking": thinking,
                    }
                )
                + "\n"
            )

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(samples)}, Accuracy: {correct}/{i + 1}")

    results = {
        "accuracy": correct / len(samples),
        "total": len(samples),
        "correct": correct,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_evaluation(Path(args.config))


if __name__ == "__main__":
    main()
