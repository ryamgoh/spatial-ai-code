"""
Analyze evaluation results across runs.
Usage:
    uv run analyze --last N
    uv run analyze --runs run_id1,run_id2
Examples:
    uv run analyze --last 2
    uv run analyze --runs run_20260221_103000,run_20260221_110000
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from models import (
    AnalysisResult,
    ErrorCategory,
    RunResult,
    TaskMetrics,
)

RESULTS_DIR = Path("results")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze evaluation results across runs"
    )
    parser.add_argument(
        "--last", type=int, default=None, help="Analyze the last N runs"
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Comma-separated list of run IDs to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for analysis results (JSON)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information"
    )
    return parser.parse_args()


def scan_results_directory() -> list[dict]:
    """Scan results/ directory and return list of run metadata."""
    runs = []

    if not RESULTS_DIR.exists():
        return runs

    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            runs.append(metadata)

    return runs


def get_runs_to_analyze(args) -> list[dict]:
    """Get the list of runs to analyze based on args."""
    all_runs = scan_results_directory()

    if not all_runs:
        print("No runs found in results/ directory.")
        sys.exit(0)

    # Sort by timestamp (newest first)
    all_runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    if args.last:
        runs = all_runs[: args.last]
        print(f"Analyzing last {len(runs)} run(s)")
    elif args.runs:
        run_ids = [r.strip() for r in args.runs.split(",")]
        runs = [r for r in all_runs if r.get("id") in run_ids]
        missing = set(run_ids) - {r.get("id") for r in runs}
        if missing:
            print(f"Warning: Run IDs not found: {missing}")
        print(f"Analyzing {len(runs)} specified run(s)")
    else:
        # Default to last 2 runs
        runs = all_runs[:2]
        print(f"No selection specified. Analyzing last {len(runs)} run(s)")

    if not runs:
        print("No runs to analyze.")
        sys.exit(0)

    return runs


def load_run_results(run: dict) -> RunResult:
    """Load results for a single run."""
    run_id = run["id"]
    results_path = Path(run.get("results_path", f"results/{run_id}")) / "results.json"

    if not results_path.exists():
        print(f"Warning: Results file not found for run {run_id}: {results_path}")
        return RunResult(
            run_id=run_id,
            model=run.get("base_model") or run.get("model", "unknown"),
            timestamp=(
                datetime.fromisoformat(run["timestamp"])
                if run.get("timestamp")
                else datetime.now()
            ),
        )

    with open(results_path) as f:
        raw_results = json.load(f)

    task_results = []
    for task_name, task_data in raw_results.items():
        if isinstance(task_data, dict) and "results" in task_data:
            metrics = task_data.get("results", {}).get(task_name, {})

            def get_metric(metrics_dict: dict, *keys: str) -> float | None:
                for key in keys:
                    if key in metrics_dict:
                        return metrics_dict[key]
                    for k in metrics_dict:
                        if k.startswith(key + ",") or k == key:
                            return metrics_dict[k]
                return None

            task_metrics = TaskMetrics(
                task_name=task_name,
                accuracy=get_metric(metrics, "acc", "accuracy"),
                f1=get_metric(metrics, "f1"),
                exact_match=get_metric(metrics, "exact_match", "em"),
            )

            known_prefixes = ("acc", "accuracy", "f1", "exact_match", "em", "mcc")
            for key, value in metrics.items():
                base_key = key.split(",")[0]
                if not any(
                    base_key.startswith(prefix) for prefix in known_prefixes
                ) and not key.endswith("_stderr"):
                    if isinstance(value, (int, float)):
                        task_metrics.custom_metrics[key] = value

            task_results.append(task_metrics)

    return RunResult(
        run_id=run_id,
        model=run.get("base_model") or run.get("model", "unknown"),
        timestamp=(
            datetime.fromisoformat(run["timestamp"])
            if run.get("timestamp")
            else datetime.now()
        ),
        task_results=task_results,
        raw_results=raw_results,
    )


def compute_summary(run_results: list[RunResult]) -> dict[str, dict[str, float | None]]:
    """Compute summary metrics per run."""
    summary = {}

    for run in run_results:
        run_summary = {}

        # Average accuracy across tasks
        accuracies = [t.accuracy for t in run.task_results if t.accuracy is not None]
        if accuracies:
            run_summary["avg_accuracy"] = sum(accuracies) / len(accuracies)

        # Average F1 across tasks
        f1_scores = [t.f1 for t in run.task_results if t.f1 is not None]
        if f1_scores:
            run_summary["avg_f1"] = sum(f1_scores) / len(f1_scores)

        # Count of tasks
        run_summary["num_tasks"] = len(run.task_results)

        summary[run.run_id] = run_summary

    return summary


def compute_per_task_breakdown(
    run_results: list[RunResult],
) -> dict[str, list[TaskMetrics]]:
    """Compute per-task metrics across all runs."""
    breakdown = defaultdict(list)

    for run in run_results:
        for task in run.task_results:
            breakdown[task.task_name].append(task)

    return dict(breakdown)


def load_responses(run: dict, task_name: str) -> list[dict]:
    """Load response JSONL for a specific task."""
    run_id = run.get("id")
    responses_path = (
        Path(run.get("results_path", f"results/{run_id}"))
        / f"responses_{task_name}.jsonl"
    )

    if not responses_path.exists():
        return []

    responses = []
    with open(responses_path) as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))

    return responses


def compute_error_analysis(
    run_results: list[RunResult], runs: list[dict]
) -> list[ErrorCategory]:
    """Analyze error patterns across runs."""
    errors_by_pattern = defaultdict(list)

    # Create a lookup for runs
    run_lookup = {r.run_id: r for r in run_results}

    for run_meta in runs:
        run_id = run_meta["id"]
        run = run_lookup.get(run_id)

        if not run:
            continue

        for task in run.task_results:
            # Check for low accuracy tasks
            if task.accuracy is not None and task.accuracy < 0.5:
                pattern = f"Low accuracy on {task.task_name}"
                errors_by_pattern[pattern].append(f"{run_id}: {task.accuracy:.2%}")

            # Load actual responses to analyze errors
            responses = load_responses(run_meta, task.task_name)
            wrong_answers = []

            for resp in responses:
                target = resp.get("target")
                filtered_resps = resp.get("filtered_resps")

                predicted = None
                num_choices = 0
                if isinstance(filtered_resps, list) and len(filtered_resps) > 0:
                    num_choices = len(filtered_resps)
                    if (
                        isinstance(filtered_resps[0], list)
                        and len(filtered_resps[0]) >= 1
                    ):
                        scores = [r[0] for r in filtered_resps]
                        predicted = scores.index(max(scores))

                labels = (
                    [chr(65 + i) for i in range(num_choices)] if num_choices > 0 else []
                )

                if predicted is not None and target is not None and predicted != target:
                    predicted_label = (
                        labels[predicted] if predicted < len(labels) else str(predicted)
                    )
                    expected_label = (
                        labels[target] if target < len(labels) else str(target)
                    )
                    wrong_answers.append(
                        {
                            "doc_id": resp.get("doc_id"),
                            "predicted": predicted,
                            "expected": target,
                            "predicted_label": predicted_label,
                            "expected_label": expected_label,
                        }
                    )

            if wrong_answers:
                pattern = f"Misclassification in {task.task_name}"
                for wa in wrong_answers[:5]:
                    errors_by_pattern[pattern].append(
                        f"{run_id} doc={wa['doc_id']}: predicted={wa['predicted_label']} (idx {wa['predicted']}), expected={wa['expected_label']} (idx {wa['expected']})"
                    )

    error_categories = []
    for pattern, examples in errors_by_pattern.items():
        error_categories.append(
            ErrorCategory(
                category=pattern,
                count=len(examples),
                examples=examples[:5],
            )
        )

    return error_categories


def find_best_run(summary: dict[str, dict[str, float | None]]) -> str | None:
    """Find the best performing run based on average accuracy."""
    best_run = None
    best_accuracy = -1

    for run_id, metrics in summary.items():
        accuracy = metrics.get("avg_accuracy", 0) or 0
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run = run_id

    return best_run


def print_analysis(analysis: AnalysisResult, verbose: bool = False) -> None:
    """Print analysis results to console."""
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    # Summary
    print("\n📊 SUMMARY")
    print("-" * 40)
    header = f"{'Run ID':<25} {'Avg Acc':>10} {'Avg F1':>10} {'Tasks':>8}"
    print(header)
    print("-" * len(header))

    for run_id, metrics in analysis.summary.items():
        avg_acc = (
            f"{metrics.get('avg_accuracy', 0):.2%}"
            if metrics.get("avg_accuracy")
            else "N/A"
        )
        avg_f1 = f"{metrics.get('avg_f1', 0):.2f}" if metrics.get("avg_f1") else "N/A"
        num_tasks = metrics.get("num_tasks", 0)
        print(f"{run_id:<25} {avg_acc:>10} {avg_f1:>10} {num_tasks:>8}")

    if analysis.best_run:
        print(f"\n🏆 Best run: {analysis.best_run}")

    # Per-task breakdown
    print("\n📋 PER-TASK BREAKDOWN")
    print("-" * 40)

    for task_name, task_metrics in analysis.per_task_breakdown.items():
        print(f"\nTask: {task_name}")
        print(f"  {'Run ID':<25} {'Accuracy':>10} {'F1':>10}")
        print(f"  {'-' * 47}")
        for tm in task_metrics:
            acc = f"{tm.accuracy:.2%}" if tm.accuracy else "N/A"
            f1 = f"{tm.f1:.2f}" if tm.f1 else "N/A"
            print(f"  {tm.task_name:<25} {acc:>10} {f1:>10}")

    # Error analysis
    if analysis.error_analysis:
        print("\n⚠️  ERROR ANALYSIS")
        print("-" * 40)

        for error in analysis.error_analysis:
            print(f"\n{error.category} ({error.count} occurrences)")
            if verbose and error.examples:
                for i, example in enumerate(error.examples[:3], 1):
                    print(f"  {i}. {example}")

    print("\n" + "=" * 60)


def save_analysis(analysis: AnalysisResult, output_path: Path) -> None:
    """Save analysis results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(analysis.model_dump(mode="json"), f, indent=2, default=str)

    print(f"Analysis saved to: {output_path}")


def main():
    # Step 1: Parse command-line arguments
    args = parse_args()

    # Step 2: Get runs to analyze (by --last or --runs)
    runs = get_runs_to_analyze(args)

    print(f"\nRuns to analyze:")
    for run in runs:
        print(
            f"  - {run['id']}: {run.get('base_model') or run.get('model')} (tasks: {run.get('tasks')})"
        )

    # Step 3: Load results for each run
    print("\nLoading results...")
    run_results = [load_run_results(run) for run in runs]

    # Step 4: Compute analysis metrics
    print("Computing analysis...")
    summary = compute_summary(run_results)
    per_task_breakdown = compute_per_task_breakdown(run_results)
    error_analysis = compute_error_analysis(run_results, runs)
    best_run = find_best_run(summary)

    # Step 5: Create analysis result object
    analysis = AnalysisResult(
        runs_compared=[r.run_id for r in run_results],
        generated_at=datetime.now(),
        summary=summary,
        per_task_breakdown=per_task_breakdown,
        error_analysis=error_analysis,
        best_run=best_run,
    )

    # Step 6: Print results to console
    print_analysis(analysis, verbose=args.verbose)

    # Step 7: Save analysis to JSON file
    if args.output:
        save_analysis(analysis, Path(args.output))
    else:
        output_path = (
            RESULTS_DIR / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        save_analysis(analysis, output_path)


if __name__ == "__main__":
    main()
