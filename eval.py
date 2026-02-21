"""
Evaluate a model using LM Eval Harness.

Usage:
    uv run eval --model <model_path> --tasks <task1,task2>

Examples:
    uv run eval --model Qwen/Qwen2.5-7B-Instruct --tasks task_a,task_b
    uv run eval --model ./outputs/qwen3-7b-lora-finetuned --tasks task_a --num-fewshot 5
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
import lm_eval
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.tasks import TaskManager

from models import RunMetadata

TASKS_DIR = Path("configs/tasks")
RESULTS_DIR = Path("results")


def generate_run_id() -> str:
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model using LM Eval Harness"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model path or HuggingFace ID"
    )
    parser.add_argument(
        "--tasks", type=str, required=True, help="Comma-separated list of task names"
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (default: task-specific)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="Batch size for vLLM (int or 'auto')",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save responses every N samples"
    )
    parser.add_argument(
        "--thinking", action="store_true", help="Enable thinking model workflow"
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        default=False,
        help="Apply chat template to prompts",
    )
    parser.add_argument(
        "--cache-requests",
        action="store_true",
        default=False,
        help="Cache preprocessed prompts for faster re-runs",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=100000,
        help="Bootstrap iterations for confidence intervals",
    )
    return parser.parse_args()


def load_task_config(task_name: str) -> dict:
    task_path = TASKS_DIR / f"{task_name}.yaml"

    if not task_path.exists():
        print(f"Error: Task config not found: {task_path}")
        print(f"Available tasks in {TASKS_DIR}:")
        for f in TASKS_DIR.glob("*.yaml"):
            print(f"  - {f.stem}")
        sys.exit(1)

    with open(task_path) as f:
        config = yaml.safe_load(f)

    return config


def validate_tasks(task_names: list[str]) -> dict[str, dict]:
    task_configs = {}
    for task in task_names:
        config = load_task_config(task)
        if "choices" not in config:
            print(f"Warning: Task '{task}' has no 'choices' field. Using default A-D.")
            config["choices"] = ["A", "B", "C", "D"]
        task_configs[task] = config
    return task_configs


def save_metadata(
    output_dir: Path,
    run_id: str,
    model: str,
    tasks: list[str],
    task_configs: dict,
    args,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": run_id,
        "timestamp": datetime.now().isoformat(),
        "type": "eval",
        "model": model,
        "tasks": tasks,
        "task_configs": {
            k: {"choices": v.get("choices")} for k, v in task_configs.items()
        },
        "results_path": str(output_dir),
        "config": {
            "num_fewshot": args.num_fewshot,
            "apply_chat_template": args.apply_chat_template,
            "seed": args.seed,
            "batch_size": args.batch_size,
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {output_dir / 'metadata.json'}")


class ResponseWriter:
    def __init__(self, output_path: Path, flush_every: int = 10):
        self.output_path = output_path
        self.flush_every = flush_every
        self.buffer: list[dict] = []
        self.total_written = 0

    def add(self, sample: dict) -> None:
        self.buffer.append(sample)
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        with open(self.output_path, "a") as f:
            for sample in self.buffer:
                f.write(json.dumps(sample, default=str) + "\n")

        self.total_written += len(self.buffer)
        self.buffer = []
        print(f"  Written {self.total_written} responses to {self.output_path.name}")

    def close(self) -> None:
        self.flush()
        print(f"  Total responses saved: {self.total_written}")


def run_evaluation(
    model_name: str,
    tasks: list[str],
    task_configs: dict,
    args,
    output_dir: Path,
) -> dict:
    print(f"Loading model: {model_name}")
    model_obj = VLLM(
        pretrained=model_name,
        guided_decoding_backend="outlines",
        batch_size=args.batch_size if isinstance(args.batch_size, int) else "auto",
    )

    task_manager = TaskManager(include_path=str(TASKS_DIR))

    all_results = {}

    for task_name in tasks:
        print(f"\nEvaluating task: {task_name}")
        print("-" * 40)

        choices = task_configs[task_name].get("choices", ["A", "B", "C", "D"])
        print(f"Choices: {choices}")

        responses_path = output_dir / f"responses_{task_name}.jsonl"
        response_writer = ResponseWriter(responses_path, flush_every=args.save_every)

        common_kwargs = {
            "model": model_obj,
            "tasks": [task_name],
            "task_manager": task_manager,
            "num_fewshot": args.num_fewshot,
            "limit": args.limit,
            "batch_size": args.batch_size,
            "apply_chat_template": args.apply_chat_template,
            "cache_requests": args.cache_requests,
            "random_seed": args.seed,
            "numpy_random_seed": args.seed,
            "torch_random_seed": args.seed,
            "fewshot_random_seed": args.seed,
            "bootstrap_iters": args.bootstrap_iters,
            "log_samples": True,
        }

        if args.thinking:
            res_think = lm_eval.simple_evaluate(**common_kwargs)

            augmented_docs = []
            for sample in res_think["samples"][task_name]:
                doc = sample["doc"]
                doc["thought"] = sample["resps"][0]
                augmented_docs.append(doc)

            results = lm_eval.simple_evaluate(
                **common_kwargs,
                gen_kwargs=f"guided_choice={choices}",
            )
        else:
            results = lm_eval.simple_evaluate(
                **common_kwargs,
                gen_kwargs=f"guided_choice={choices}",
            )

        if "samples" in results and task_name in results["samples"]:
            samples = results["samples"][task_name]
            for sample in samples:
                response_record = {
                    "doc_id": sample.get("doc_id"),
                    "doc": sample.get("doc"),
                    "target": sample.get("target"),
                    "response": sample.get("resps", [[]])[0]
                    if sample.get("resps")
                    else None,
                    "filtered_resps": sample.get("filtered_resps"),
                }
                response_writer.add(response_record)
            response_writer.close()

        if task_name in results.get("results", {}):
            task_results = results["results"][task_name]
            print(f"Results for {task_name}:")
            for metric, value in task_results.items():
                if not metric.endswith("_stderr"):
                    print(f"  {metric}: {value}")

        all_results[task_name] = results

    del model_obj

    return all_results


def save_results(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_only = {}
    for task_name, task_data in results.items():
        if isinstance(task_data, dict):
            metrics_only[task_name] = {
                "results": task_data.get("results", {}),
            }

    with open(output_dir / "results.json", "w") as f:
        json.dump(metrics_only, f, indent=2, default=str)

    print(f"Metrics saved to: {output_dir / 'results.json'}")


def main():
    args = parse_args()

    task_names = [t.strip() for t in args.tasks.split(",")]
    task_configs = validate_tasks(task_names)

    run_id = generate_run_id()
    output_dir = RESULTS_DIR / run_id

    print("=" * 50)
    print(f"Run ID: {run_id}")
    print(f"Model: {args.model}")
    print(f"Tasks: {task_names}")
    print(f"Few-shot: {args.num_fewshot or 'task-default'}")
    print(f"Chat template: {args.apply_chat_template}")
    print(f"Cache requests: {args.cache_requests}")
    print(f"Seed: {args.seed}")
    print(f"Thinking mode: {args.thinking}")
    for task, config in task_configs.items():
        print(f"  {task} choices: {config.get('choices')}")
    print("=" * 50)

    save_metadata(output_dir, run_id, args.model, task_names, task_configs, args)

    results = run_evaluation(
        model_name=args.model,
        tasks=task_names,
        task_configs=task_configs,
        args=args,
        output_dir=output_dir,
    )

    save_results(results, output_dir)

    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print(f"Results directory: {output_dir}")
    print(f"  - metadata.json")
    print(f"  - results.json")
    for task in task_names:
        print(f"  - responses_{task}.jsonl")
    print("=" * 50)


if __name__ == "__main__":
    main()
