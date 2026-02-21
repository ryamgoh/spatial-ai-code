from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class RunMetadata(BaseModel):
    """Metadata for a single run (finetune or eval)."""

    id: str = Field(description="Run ID, e.g., run_20260221_103000")
    timestamp: datetime = Field(description="When the run was created")
    type: Literal["finetune", "eval"] = Field(description="Type of run")
    model: str = Field(description="Model path or HuggingFace ID")
    tasks: list[str] | None = Field(default=None, description="Tasks for eval runs")
    config: str | None = Field(
        default=None, description="Config name for finetune runs"
    )
    results_path: str = Field(description="Path to results directory")


class TaskMetrics(BaseModel):
    """Metrics for a single task evaluation."""

    task_name: str
    accuracy: float | None = None
    f1: float | None = None
    exact_match: float | None = None
    bleu: float | None = None
    rouge: float | None = None
    custom_metrics: dict[str, float] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Results from a single eval run."""

    run_id: str
    model: str
    timestamp: datetime
    task_results: list[TaskMetrics] = Field(default_factory=list)
    raw_results: dict[str, Any] = Field(default_factory=dict)


class ErrorCategory(BaseModel):
    """Categorized error information."""

    category: str
    count: int
    examples: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Detailed analysis result comparing runs."""

    runs_compared: list[str]
    generated_at: datetime = Field(default_factory=datetime.now)

    # Summary comparison
    summary: dict[str, dict[str, float | None]] = Field(
        default_factory=dict, description="Per-run summary: {run_id: {metric: value}}"
    )

    # Per-task breakdown
    per_task_breakdown: dict[str, list[TaskMetrics]] = Field(
        default_factory=dict, description="Task name -> list of TaskMetrics per run"
    )

    # Error analysis
    error_analysis: list[ErrorCategory] = Field(
        default_factory=list, description="Categorized error patterns"
    )

    # Best performer
    best_run: str | None = Field(
        default=None, description="Run ID with best overall performance"
    )
