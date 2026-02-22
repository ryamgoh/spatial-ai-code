"""Pydantic models for run tracking and configuration."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RAGCorpusConfig(BaseModel):
    paths: list[str] = Field(default_factory=list)
    chunk_size: int = 800
    chunk_overlap: int = 100


class RAGConfig(BaseModel):
    enabled: bool = False
    context_k: int = 3
    context_template: str = "- {text}"
    context_separator: str = "\n"
    query_field: str = "text"
    context_field: str = "context"
    corpus: RAGCorpusConfig | None = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class TaskConfig(BaseModel):
    task: str
    choices: list[str] = Field(default_factory=lambda: ["A", "B", "C", "D"])
    rag: RAGConfig | None = None


class RunMetadata(BaseModel):
    id: str
    timestamp: datetime
    type: Literal["finetune", "eval"]
    model: str
    tasks: list[str] | None = None
    config: str | None = None
    results_path: str


class TaskMetrics(BaseModel):
    task_name: str
    accuracy: float | None = None
    f1: float | None = None
    exact_match: float | None = None
    custom_metrics: dict[str, float] = Field(default_factory=dict)


class RunResult(BaseModel):
    run_id: str
    model: str
    timestamp: datetime
    task_results: list[TaskMetrics] = Field(default_factory=list)
    raw_results: dict[str, Any] = Field(default_factory=dict)


class ErrorCategory(BaseModel):
    category: str
    count: int
    examples: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    runs_compared: list[str]
    generated_at: datetime = Field(default_factory=datetime.now)
    summary: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    per_task_breakdown: dict[str, list[TaskMetrics]] = Field(default_factory=dict)
    error_analysis: list[ErrorCategory] = Field(default_factory=list)
    best_run: str | None = None
