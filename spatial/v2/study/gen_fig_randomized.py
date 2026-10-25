"""Create four research figures from analyze_randomized CSVs (PDF and PNG)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = [
    "untuned", "single-natural-seed-42", "single-natural-seed-43",
    "single-answer-only-seed-42", "single-answer-only-seed-43",
    "multi-natural-seed-42", "multi-natural-seed-43",
]
LABELS = ["Untuned", "SINGLE Natural 42", "SINGLE Natural 43",
          "SINGLE answer-only 42", "SINGLE answer-only 43",
          "MULTI Natural 42", "MULTI Natural 43"]
COLORS = ["#777777", "#0072B2", "#0072B2", "#E69F00", "#E69F00", "#009E73", "#009E73"]
CONTRACTS = ["single", "all-possible"]
PARTITIONS = ["matched", "length-ood", "noise-ood", "composition-ood"]


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save(fig, output, name):
    for extension in ("pdf", "png"):
        fig.savefig(output / f"fig_{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    output = args.analysis / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })
    contracts = read(args.analysis / "contracts.csv")
    lookup = {(r["model"], r["contract"]): r for r in contracts if r["panel"] == "sft"}
    matrix = np.array([[float(lookup[m, c]["strict_acc"]) for c in CONTRACTS] for m in MODELS])
    fig, ax = plt.subplots(figsize=(6.75, 3.4), layout="constrained")
    image = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    for i in range(len(MODELS)):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] < 55 else "black", fontsize=9)
    ax.set_yticks(range(len(MODELS)), LABELS)
    ax.set_xticks([0, 1], ["SINGLE (n=900)", "ALL-POSSIBLE (n=2,700)"])
    ax.set_title("Contract transfer: frozen two-stage accuracy")
    fig.colorbar(image, ax=ax, label="Exact answer-set accuracy (%)", shrink=.85)
    save(fig, output, "contract_transfer")

    partitions = read(args.analysis / "partitions.csv")
    lookup = {(r["model"], r["contract"], r["partition"]): r for r in partitions if r["panel"] == "sft"}
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.2), layout="constrained")
    for ax, contract in zip(axes, CONTRACTS):
        for i, model in enumerate(MODELS):
            y = [float(lookup[model, contract, p]["strict_acc"]) for p in PARTITIONS]
            ax.plot(range(4), y, color=COLORS[i], marker="s" if model.endswith("43") else "o",
                    linestyle="--" if model.endswith("43") else "-", linewidth=1.3,
                    markersize=3, label=LABELS[i])
        ax.set_xticks(range(4), ["Matched", "Length", "Noise", "Composition"], rotation=25)
        ax.set_ylim(-3, 103)
        ax.set_title(contract.upper())
        ax.set_ylabel("Exact accuracy (%)")
        ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    save(fig, output, "partition_comparison")

    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.4), layout="constrained")
    for row_index, (filename, dimension, xlabel) in enumerate(
        [("length", "L", "Verified semantic length L"), ("noise", "X", "Distractor premises X")]
    ):
        data = read(args.analysis / f"{filename}.csv")
        for col, contract in enumerate(CONTRACTS):
            ax = axes[row_index, col]
            for i, model in enumerate(MODELS):
                group = sorted(
                    [r for r in data if r["panel"] == "sft" and r["model"] == model
                     and r["contract"] == contract], key=lambda r: int(r[dimension])
                )
                x = [int(r[dimension]) for r in group]
                ax.plot(x, [float(r["strict_acc"]) for r in group],
                        color=COLORS[i], marker="s" if model.endswith("43") else "o",
                        linestyle="--" if model.endswith("43") else "-",
                        linewidth=1.3, markersize=3, label=LABELS[i])
            ax.set_title(contract.upper())
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Exact accuracy (%)")
            ax.set_ylim(-3, 103)
            ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    save(fig, output, "structural_curves")

    interface = read(args.analysis / "interface.csv")
    fig, ax = plt.subplots(figsize=(6.75, 5.7), layout="constrained")
    for i, r in enumerate(interface):
        a, b = float(r["stage1_acc"]), float(r["strict_acc"])
        ax.plot([a, b], [i, i], color="#aaaaaa", linewidth=1)
        ax.scatter(a, i, color="#0072B2", marker="o", s=22)
        ax.scatter(b, i, color="#D55E00", marker="s", s=22)
        ax.scatter(float(r["normalized_stage1_acc"]), i, color="#009E73",
                   marker="|", s=75)
    ax.scatter([], [], color="#0072B2", marker="o", label="Native stage 1")
    ax.scatter([], [], color="#D55E00", marker="s", label="Stage 2")
    ax.scatter([], [], color="#009E73", marker="|", label="Normalized footer only")
    ax.set_yticks(range(len(interface)), [r["model"] for r in interface], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Exact answer-set accuracy (%)")
    ax.set_title("Scoring interface: same examples, different response stages")
    ax.grid(axis="x", alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3)
    save(fig, output, "stage_comparison")
    print(f"Exported four PDF/PNG figures to {output}")


if __name__ == "__main__":
    main()
