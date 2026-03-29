"""
Create grouped bar chart visualizations from evaluation results.json files.

Usage:
    uv run create_visual eval/results/run1/results.json eval/results/run2/results.json
    uv run create_visual eval/results/*/results.json --labels "Baseline,RAG"
    uv run create_visual eval/results/*/results.json --metrics acc_gen,strict_acc
    uv run create_visual eval/results/*/results.json --output-dir eval/charts/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create grouped bar charts from evaluation results"
    )
    parser.add_argument(
        "results_files",
        nargs="+",
        type=str,
        help="Paths to results.json files",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated metric names to plot (default: all non-stderr metrics)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated display labels for each results file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="charts",
        help="Output directory for charts (default: charts)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image DPI (default: 150)",
    )
    return parser.parse_args()


def shorten_task_name(task_name: str) -> str:
    """Strip common prefixes and replace underscores with spaces."""
    name = task_name
    for prefix in ("spatial_eval_gen_", "spatial_eval_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", " ")


def extract_label(data: dict) -> str:
    """Extract display label from configs.*.metadata."""
    configs = data.get("configs", {})
    for task_name, task_config in configs.items():
        metadata = task_config.get("metadata", {})
        model = metadata.get("model_type") or metadata.get("pretrained", "")
        note = metadata.get("notes", "")
        if model:
            # Shorten long model paths
            model = model.split("/")[-1]
            if note:
                return f"{model} ({notes})"
            return model
        if note:
            return note
    return "unknown"


def extract_metrics(data: dict) -> dict[str, dict[str, dict[str, float]]]:
    """
    Extract metrics from results data.

    Returns:
        {task_name: {metric_name: {"value": float, "stderr": float or None}}}
    """
    results = data.get("results", {})
    task_metrics = {}

    non_metric_keys = {"name", "alias", "sample_len"}

    for task_name, task_data in results.items():
        if not isinstance(task_data, dict):
            continue

        metrics = {}
        stderr_keys = {}

        for key, value in task_data.items():
            if not isinstance(value, (int, float)):
                continue
            if key in non_metric_keys:
                continue

            # Split on first comma: "metric_name,filter" -> ("metric_name", "filter")
            if "," in key:
                base_key = key.split(",")[0]
            else:
                base_key = key

            if base_key.endswith("_stderr"):
                # Store stderr separately, mapped to the base metric name
                base_metric = base_key[: -len("_stderr")]
                stderr_keys[base_metric] = value
            else:
                metrics[base_key] = {"value": value, "stderr": None}

        # Merge stderr values into metrics
        for metric_name, stderr_val in stderr_keys.items():
            if metric_name in metrics:
                metrics[metric_name]["stderr"] = stderr_val

        if metrics:
            task_metrics[task_name] = metrics

    return task_metrics


def get_colors(n: int) -> list:
    """Get a list of distinct colors for bars."""
    cmap = matplotlib.colormaps.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def plot_metric(
    metric_name: str,
    all_data: list[dict],
    output_path: Path,
    dpi: int,
):
    """
    Plot a grouped bar chart for a single metric.

    Args:
        metric_name: Name of the metric (e.g., "acc_gen")
        all_data: List of dicts with keys "label", "tasks" (task->value/stderr)
        output_path: Where to save the figure
        dpi: Image resolution
    """
    # Collect all tasks across all models
    all_tasks = []
    seen = set()
    for run in all_data:
        for task in run["tasks"]:
            if task not in seen:
                all_tasks.append(task)
                seen.add(task)

    if not all_tasks:
        return

    n_runs = len(all_data)
    n_tasks = len(all_tasks)
    x = np.arange(n_tasks)
    width = 0.8 / n_runs
    colors = get_colors(n_runs)

    fig, ax = plt.subplots(figsize=(max(6, n_tasks * 2), 5))

    for i, run in enumerate(all_data):
        values = []
        stderrs = []
        for task in all_tasks:
            info = run["tasks"].get(task, {})
            values.append(info.get("value", 0) * 100)
            stderrs.append(info.get("stderr", 0) * 100 if info.get("stderr") else 0)

        offset = (i - (n_runs - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            yerr=stderrs if any(s > 0 for s in stderrs) else None,
            label=run["label"],
            color=colors[i],
            capsize=3,
            error_kw={"linewidth": 1},
        )

        # Add percentage labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stderrs) * 0.5 + 1,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Task")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(metric_name.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(
        [shorten_task_name(t) for t in all_tasks],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_overview(
    metric_names: list[str],
    all_data: list[dict],
    output_path: Path,
    dpi: int,
):
    """Plot all metrics as subplots in a single overview figure."""
    n_metrics = len(metric_names)
    if n_metrics == 0:
        return

    cols = min(n_metrics, 2)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(max(8, cols * 6), rows * 5), squeeze=False
    )

    for idx, metric_name in enumerate(metric_names):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        # Collect tasks for this metric
        all_tasks = []
        seen = set()
        for run in all_data:
            if metric_name in run["by_metric"]:
                for task in run["by_metric"][metric_name]:
                    if task not in seen:
                        all_tasks.append(task)
                        seen.add(task)

        if not all_tasks:
            ax.set_visible(False)
            continue

        n_runs = len(all_data)
        n_tasks = len(all_tasks)
        x = np.arange(n_tasks)
        width = 0.8 / n_runs
        colors = get_colors(n_runs)

        for i, run in enumerate(all_data):
            metric_tasks = run["by_metric"].get(metric_name, {})
            values = []
            stderrs = []
            for task in all_tasks:
                info = metric_tasks.get(task, {})
                values.append(info.get("value", 0) * 100)
                stderrs.append(info.get("stderr", 0) * 100 if info.get("stderr") else 0)

            offset = (i - (n_runs - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                yerr=stderrs if any(s > 0 for s in stderrs) else None,
                label=run["label"],
                color=colors[i],
                capsize=2,
                error_kw={"linewidth": 0.8},
            )

            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{val:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )

        ax.set_title(metric_name.replace("_", " ").title(), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [shorten_task_name(t) for t in all_tasks],
            rotation=30,
            ha="right",
            fontsize=7,
        )
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)
        ax.set_ylabel("Accuracy (%)", fontsize=8)

    # Hide unused subplots
    for idx in range(n_metrics, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Evaluation Results Overview", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()

    # Load all results files
    runs = []
    labels = args.labels.split(",") if args.labels else []

    for i, filepath in enumerate(args.results_files):
        path = Path(filepath)
        if not path.exists():
            print(f"Warning: File not found: {filepath}", file=sys.stderr)
            continue

        with open(path) as f:
            data = json.load(f)

        label = labels[i] if i < len(labels) else extract_label(data)
        metrics = extract_metrics(data)

        # Build by_metric lookup: {metric_name: {task_name: {value, stderr}}}
        by_metric = {}
        for task_name, task_metrics in metrics.items():
            for metric_name, info in task_metrics.items():
                by_metric.setdefault(metric_name, {})[task_name] = info

        runs.append({"label": label, "tasks_metrics": metrics, "by_metric": by_metric})

    if not runs:
        print("No valid results files found.", file=sys.stderr)
        sys.exit(1)

    # Determine which metrics to plot
    all_metric_names = set()
    for run in runs:
        all_metric_names.update(run["by_metric"].keys())

    if args.metrics:
        selected_metrics = [m.strip() for m in args.metrics.split(",")]
    else:
        selected_metrics = sorted(all_metric_names)

    # Filter out metrics that are all zeros
    selected_metrics = [
        m
        for m in selected_metrics
        if any(
            run["by_metric"].get(m, {}).get(t, {}).get("value", 0) > 0
            for run in runs
            for t in run["by_metric"].get(m, {})
        )
    ]

    if not selected_metrics:
        print("No metrics with non-zero values found.", file=sys.stderr)
        sys.exit(1)

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Restructure data for plotting
    all_data = []
    for run in runs:
        all_data.append({"label": run["label"], "by_metric": run["by_metric"]})

    # Generate individual metric charts
    for metric_name in selected_metrics:
        metric_data = []
        for run in all_data:
            tasks = run["by_metric"].get(metric_name, {})
            if tasks:
                metric_data.append({"label": run["label"], "tasks": tasks})

        if metric_data:
            out_path = output_dir / f"{metric_name}.{args.format}"
            plot_metric(metric_name, metric_data, out_path, args.dpi)
            print(f"Saved: {out_path}")

    # Generate overview figure
    overview_path = output_dir / f"overview.{args.format}"
    plot_overview(selected_metrics, all_data, overview_path, args.dpi)
    print(f"Saved: {overview_path}")


if __name__ == "__main__":
    main()
