#!/usr/bin/env python3
"""Prepare and submit the V2 ablation Slurm dependency graph."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ResourceSpec:
    partition: str
    cpus: int
    memory: str
    time: str
    gres: str | None = None


@dataclass(frozen=True)
class ArmSpec:
    name: str
    view: str
    scope: str
    cell: str | None
    train_overrides: dict[str, Any]
    eval_overrides: dict[str, Any]


@dataclass(frozen=True)
class RunSpec:
    name: str
    source: Path
    matrix: Path
    data_output: Path
    base_model: str
    train_template: Path
    eval_template: Path
    train_overrides: dict[str, Any]
    eval_overrides: dict[str, Any]
    task_template: Path
    retention_task: str
    arms: tuple[ArmSpec, ...]
    resources: dict[str, ResourceSpec]


@dataclass(frozen=True)
class PreparedArm:
    name: str
    train_view: Path
    test_view: Path
    train_config: Path
    eval_config: Path
    task_config: Path
    model_dir: Path
    result_dir: Path


@dataclass(frozen=True)
class PreparedRun:
    run_dir: Path
    data_output: Path
    arms: tuple[PreparedArm, ...]


@dataclass(frozen=True)
class JobPlan:
    stage: str
    arm: str | None
    command: tuple[str, ...]
    job_id: str | None = None


@dataclass(frozen=True)
class SubmissionResult:
    jobs: tuple[JobPlan, ...]
    generate_job: str | None
    arms: dict[str, dict[str, str]]
    summary_job: str | None


Runner = Callable[[list[str]], str]


@dataclass(frozen=True)
class SubmissionOptions:
    run_id: str
    run_root: Path | None = None
    dry_run: bool = False
    only: tuple[str, ...] | None = None
    skip_generation: bool = False
    replace_data: bool = False
    resume: bool = False


@dataclass
class SubmissionState:
    dry_run: bool
    runner: Runner
    jobs_file: Path
    jobs: list[JobPlan] = field(default_factory=list)
    submitted: list[str] = field(default_factory=list)
    generate_job: str | None = None
    arm_jobs: dict[str, dict[str, str]] = field(default_factory=dict)
    summary_job: str | None = None

    def issue(self, stage: str, arm: str | None, command: list[str]) -> str:
        if self.dry_run:
            job_id = f"<{stage}{':' + arm if arm else ''}>"
            self.jobs.append(JobPlan(stage, arm, tuple(command)))
        else:
            try:
                job_id = self.runner(command)
            except Exception as exc:
                self.persist()
                cancel = (
                    "scancel " + " ".join(self.submitted)
                    if self.submitted
                    else "nothing submitted"
                )
                raise RuntimeError(
                    f"failed to submit {stage}{':' + arm if arm else ''}; {cancel}"
                ) from exc
            self.jobs.append(JobPlan(stage, arm, tuple(command), job_id))
        self.submitted.append(job_id)
        if stage == "generate":
            self.generate_job = job_id
        elif stage in {"train", "eval"}:
            assert arm is not None
            self.arm_jobs.setdefault(arm, {})[stage] = job_id
        elif stage == "summarize":
            self.summary_job = job_id
        if not self.dry_run:
            self.persist()
        return job_id

    def persist(self) -> None:
        _write_jobs(
            self.jobs_file,
            self.generate_job,
            self.arm_jobs,
            self.summary_job,
        )

    def result(self) -> SubmissionResult:
        return SubmissionResult(
            tuple(self.jobs),
            self.generate_job,
            self.arm_jobs,
            self.summary_job,
        )


def _safe_name(value: Any, field: str) -> str:
    name = str(value or "").strip()
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(f"{field} is not a safe name: {value!r}")
    return name


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")  # noqa: TRY004
    return value


def _resolve(base: Path, value: Any, field: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{field} is required")
    path = Path(raw)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _resource(value: Any, stage: str) -> ResourceSpec:
    raw = _mapping(value, f"resources.{stage}")
    required = {"partition", "cpus", "memory", "time"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"resources.{stage} missing: {', '.join(missing)}")
    unknown = set(raw) - required - {"gres"}
    if unknown:
        raise ValueError(
            f"resources.{stage} unknown keys: {', '.join(sorted(unknown))}"
        )
    cpus = int(raw["cpus"])
    if cpus <= 0:
        raise ValueError(f"resources.{stage}.cpus must be positive")
    return ResourceSpec(
        partition=str(raw["partition"]),
        cpus=cpus,
        memory=str(raw["memory"]),
        time=str(raw["time"]),
        gres=str(raw["gres"]) if raw.get("gres") else None,
    )


def _arms(value: Any) -> tuple[ArmSpec, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("arms must be a non-empty list")
    arms = []
    for item in value:
        item = _mapping(item, "arm")
        unknown = set(item) - {
            "name",
            "view",
            "scope",
            "cell",
            "train_overrides",
            "eval_overrides",
        }
        if unknown:
            raise ValueError("unknown arm keys: " + ", ".join(sorted(unknown)))
        scope = str(item.get("scope") or "by_variant")
        if scope not in {"by_variant", "by_cell"}:
            raise ValueError("arm scope must be by_variant or by_cell")
        cell = _safe_name(item.get("cell"), "arm cell") if item.get("cell") else None
        if (scope == "by_cell") != (cell is not None):
            raise ValueError(f"{scope} arm has invalid cell setting")
        arms.append(
            ArmSpec(
                _safe_name(item.get("name"), "arm name"),
                _safe_name(item.get("view"), "arm view"),
                scope,
                cell,
                _mapping(item.get("train_overrides", {}), "arm.train_overrides"),
                _mapping(item.get("eval_overrides", {}), "arm.eval_overrides"),
            )
        )
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError("duplicate arm names")
    return tuple(arms)


def load_run_spec(path: str | Path, repo_root: str | Path = REPO_ROOT) -> RunSpec:
    """Load a strict version-1 orchestration specification."""
    source = Path(path).resolve()
    root = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "run spec")
    allowed = {
        "version",
        "name",
        "matrix",
        "data_output",
        "base_model",
        "train",
        "eval",
        "retention_task",
        "arms",
        "resources",
    }
    unknown = sorted(set(root) - allowed)
    if unknown:
        raise ValueError("unknown run keys: " + ", ".join(unknown))
    if root.get("version") != 1:
        raise ValueError("run version must be 1")
    base = source.parent
    repo_root = Path(repo_root).resolve()
    arms = _arms(root.get("arms"))
    resources_raw = _mapping(root.get("resources"), "resources")
    required_stages = ("generate", "train", "eval", "summarize")
    if set(resources_raw) != set(required_stages):
        raise ValueError("resources must define generate, train, eval, and summarize")
    train = _mapping(root.get("train"), "train")
    evaluation = _mapping(root.get("eval"), "eval")
    for name, section in (("train", train), ("eval", evaluation)):
        allowed_section = (
            {"template", "overrides", "task_template"}
            if name == "eval"
            else {"template", "overrides"}
        )
        unknown_section = set(section) - allowed_section
        if unknown_section:
            raise ValueError(
                f"unknown {name} keys: {', '.join(sorted(unknown_section))}"
            )
    spec = RunSpec(
        name=_safe_name(root.get("name"), "run name"),
        source=source,
        matrix=_resolve(base, root.get("matrix"), "matrix"),
        data_output=_resolve(repo_root, root.get("data_output"), "data_output"),
        base_model=str(root.get("base_model") or "").strip(),
        train_template=_resolve(base, train.get("template"), "train.template"),
        eval_template=_resolve(base, evaluation.get("template"), "eval.template"),
        train_overrides=_mapping(train.get("overrides", {}), "train.overrides"),
        eval_overrides=_mapping(evaluation.get("overrides", {}), "eval.overrides"),
        task_template=_resolve(
            base, evaluation.get("task_template"), "eval.task_template"
        ),
        retention_task=str(root.get("retention_task") or "").strip(),
        arms=arms,
        resources={
            stage: _resource(resources_raw[stage], stage) for stage in required_stages
        },
    )
    if not spec.base_model:
        raise ValueError("base_model is required")
    if not spec.retention_task:
        raise ValueError("retention_task is required")
    required_files = (
        spec.matrix,
        spec.train_template,
        spec.eval_template,
        spec.task_template,
    )
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "run inputs are missing: " + ", ".join(str(path) for path in missing_files)
        )
    _validate_overrides(spec)
    return spec


def _relative(path: Path, workdir: Path) -> str:
    return os.path.relpath(path, workdir)


def _deep_merge(*values: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        for key, item in value.items():
            if isinstance(item, dict) and isinstance(merged.get(key), dict):
                merged[key] = _deep_merge(merged[key], item)
            else:
                merged[key] = item
    return merged


def _validate_overrides(spec: RunSpec) -> None:
    protected_train = {"base_model", "datasets", "test_datasets", "output_dir"}
    protected_eval = {"include_path", "tasks"}
    for label, overrides, protected in (
        ("train.overrides", spec.train_overrides, protected_train),
        ("eval.overrides", spec.eval_overrides, protected_eval),
        *(
            (f"arm {arm.name}.train_overrides", arm.train_overrides, protected_train)
            for arm in spec.arms
        ),
        *(
            (f"arm {arm.name}.eval_overrides", arm.eval_overrides, protected_eval)
            for arm in spec.arms
        ),
    ):
        forbidden = sorted(set(overrides) & protected)
        if forbidden:
            raise ValueError(f"{label} cannot override: {', '.join(forbidden)}")
    for label, overrides in (
        ("eval.overrides", spec.eval_overrides),
        *((f"arm {arm.name}.eval_overrides", arm.eval_overrides) for arm in spec.arms),
    ):
        model_args = overrides.get("model_args", {})
        forbidden = sorted(set(model_args) & {"pretrained", "lora_path", "choices"})
        if forbidden:
            raise ValueError(
                f"{label}.model_args cannot override: {', '.join(forbidden)}"
            )


def _prepared_layout(
    spec: RunSpec,
    run_id: str,
    run_root: Path,
) -> PreparedRun:
    run_dir = run_root / run_id
    view_root = spec.data_output.with_suffix("").with_name(
        spec.data_output.with_suffix("").name + "_views"
    )

    def views(arm: ArmSpec) -> tuple[Path, Path]:
        directory = view_root / arm.scope
        if arm.cell:
            directory /= arm.cell
        return (
            directory / f"{arm.view}_train.jsonl",
            directory / f"{arm.view}_test.jsonl",
        )

    def prepare_arm(arm: ArmSpec) -> PreparedArm:
        train_view, test_view = views(arm)
        return PreparedArm(
            name=arm.name,
            train_view=train_view,
            test_view=test_view,
            train_config=run_dir / "configs" / f"train-{arm.name}.yaml",
            eval_config=run_dir / "configs" / f"eval-{arm.name}.yaml",
            task_config=run_dir / "tasks" / f"spatial_v2_{arm.name}.yaml",
            model_dir=run_dir / "models" / arm.name,
            result_dir=run_dir / "results" / arm.name,
        )

    arms = tuple(prepare_arm(arm) for arm in spec.arms)
    return PreparedRun(run_dir, spec.data_output, arms)


def prepare_run(
    spec: RunSpec,
    repo_root: str | Path,
    run_id: str,
    *,
    run_root: str | Path | None = None,
) -> PreparedRun:
    """Render immutable arm-specific training and evaluation configs."""
    repo_root = Path(repo_root).resolve()
    root = Path(run_root).resolve() if run_root else spec.source.parent / "runs"
    prepared = _prepared_layout(spec, _safe_name(run_id, "run id"), root)
    if prepared.run_dir.exists():
        raise FileExistsError(f"run directory already exists: {prepared.run_dir}")
    for directory in ("configs", "tasks", "models", "results", "logs"):
        (prepared.run_dir / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec.source, prepared.run_dir / "run.yaml")
    shutil.copy2(spec.matrix, prepared.run_dir / "matrix.yaml")
    shutil.copy2(
        repo_root / "experiments/tasks/utils.py", prepared.run_dir / "tasks/utils.py"
    )
    retention_source = repo_root / "experiments/tasks" / f"{spec.retention_task}.yaml"
    if not retention_source.exists():
        raise FileNotFoundError(f"retention task not found: {retention_source}")
    shutil.copy2(retention_source, prepared.run_dir / "tasks" / retention_source.name)

    train_template = yaml.safe_load(spec.train_template.read_text(encoding="utf-8"))
    eval_template = yaml.safe_load(spec.eval_template.read_text(encoding="utf-8"))
    task_template = spec.task_template.read_text(encoding="utf-8")
    finetune_dir = repo_root / "finetune"
    eval_dir = repo_root / "eval"
    choices = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    arm_specs = {arm.name: arm for arm in spec.arms}
    for arm in prepared.arms:
        arm_spec = arm_specs[arm.name]
        train = _deep_merge(
            train_template,
            spec.train_overrides,
            arm_spec.train_overrides,
        )
        train["base_model"] = spec.base_model
        train["output_dir"] = _relative(arm.model_dir, finetune_dir)
        train["datasets"] = [
            {
                "path": _relative(arm.train_view, finetune_dir),
                "type": "chat_template",
                "field_messages": "messages",
            }
        ]
        train["test_datasets"] = [
            {
                "path": _relative(arm.test_view, finetune_dir),
                "type": "chat_template",
                "field_messages": "messages",
            }
        ]
        arm.train_config.write_text(
            yaml.safe_dump(train, sort_keys=False), encoding="utf-8"
        )

        task_name = f"spatial_v2_{run_id}_{arm.name}".replace("-", "_")
        arm.task_config.write_text(
            task_template.replace("__TASK_NAME__", task_name).replace(
                "__TEST_PATH__", _relative(arm.test_view, eval_dir)
            ),
            encoding="utf-8",
        )
        evaluation = _deep_merge(
            eval_template,
            spec.eval_overrides,
            arm_spec.eval_overrides,
        )
        evaluation["model_args"] = dict(evaluation["model_args"])
        evaluation["model_args"]["pretrained"] = spec.base_model
        evaluation["model_args"]["lora_path"] = _relative(arm.model_dir, eval_dir)
        evaluation["model_args"]["choices"] = choices
        evaluation["include_path"] = _relative(prepared.run_dir / "tasks", eval_dir)
        evaluation["tasks"] = [task_name, spec.retention_task]
        arm.eval_config.write_text(
            yaml.safe_dump(evaluation, sort_keys=False), encoding="utf-8"
        )
    return prepared


def _validate_prepared_run(prepared: PreparedRun) -> None:
    missing = [
        path
        for arm in prepared.arms
        for path in (arm.train_config, arm.eval_config, arm.task_config)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "existing run is incomplete: " + ", ".join(str(path) for path in missing)
        )


def _sbatch_command(
    repo_root: Path,
    script: Path,
    resource: ResourceSpec,
    *,
    job_name: str,
    log_dir: Path,
    exports: dict[str, str],
    dependency: str | None = None,
) -> list[str]:
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--partition={resource.partition}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={resource.cpus}",
        f"--mem={resource.memory}",
        f"--time={resource.time}",
        f"--chdir={repo_root}",
        f"--output={log_dir}/%x-%j.out",
        f"--error={log_dir}/%x-%j.err",
    ]
    if resource.gres:
        command.append(f"--gres={resource.gres}")
    if dependency:
        command.extend((f"--dependency={dependency}", "--kill-on-invalid-dep=yes"))
    exported = ["ALL", *(f"{key}={value}" for key, value in sorted(exports.items()))]
    command.extend((f"--export={','.join(exported)}", str(script)))
    return command


def _default_runner(command: list[str]) -> str:
    if shutil.which("sbatch") is None:
        raise RuntimeError("sbatch is unavailable; submit from the Slurm login server")
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout.strip().split(";", 1)[0]


def _write_jobs(
    path: Path, generate: str | None, arms: dict, summary: str | None
) -> None:
    payload = {
        "schema": "spatial-v2-slurm-jobs",
        "run_id": path.parent.name,
        "generate": generate,
        "arms": arms,
        "summary": summary,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def submit_experiment(
    spec_path: str | Path,
    repo_root: str | Path,
    options: SubmissionOptions,
    *,
    runner: Runner | None = None,
) -> SubmissionResult:
    """Prepare and optionally submit one independent-job Slurm DAG."""
    repo_root = Path(repo_root).resolve()
    spec = load_run_spec(spec_path, repo_root)
    unknown = sorted(set(options.only or ()) - {arm.name for arm in spec.arms})
    if unknown:
        raise ValueError("unknown arms: " + ", ".join(unknown))
    selected = tuple(
        arm for arm in spec.arms if options.only is None or arm.name in options.only
    )
    if not selected:
        raise ValueError("no arms selected")
    if options.skip_generation and options.replace_data:
        raise ValueError("replace_data cannot be used with skip_generation")
    if not options.dry_run and runner is None and shutil.which("sbatch") is None:
        raise RuntimeError("sbatch is unavailable; submit from the Slurm login server")
    root = (
        options.run_root.resolve() if options.run_root else spec.source.parent / "runs"
    )
    run_id = _safe_name(options.run_id, "run id")
    predicted = _prepared_layout(spec, run_id, root)
    selected_names = {item.name for item in selected}
    predicted_selected = tuple(
        arm for arm in predicted.arms if arm.name in selected_names
    )
    if options.skip_generation and not options.dry_run:
        _validate_existing_views(predicted_selected, predicted.data_output)
    if options.dry_run:
        prepared = predicted
    elif options.resume and predicted.run_dir.exists():
        prepared = predicted
        _validate_prepared_run(prepared)
    else:
        prepared = prepare_run(spec, repo_root, run_id, run_root=root)
    selected_prepared = tuple(
        arm for arm in prepared.arms if arm.name in selected_names
    )
    command_runner = runner or _default_runner
    jobs_file = prepared.run_dir / "jobs.json"
    if not options.dry_run and options.resume and jobs_file.exists():
        history = prepared.run_dir / "jobs-history.jsonl"
        with history.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(json.loads(jobs_file.read_text()), sort_keys=True) + "\n"
            )
    state = SubmissionState(options.dry_run, command_runner, jobs_file)

    scripts = spec.source.parent / "jobs"
    logs = prepared.run_dir / "logs"
    if not options.skip_generation:
        state.issue(
            "generate",
            None,
            _sbatch_command(
                repo_root,
                scripts / "generate.sh",
                spec.resources["generate"],
                job_name=f"v2-gen-{run_id}",
                log_dir=logs,
                exports={
                    "MATRIX": str(spec.matrix),
                    "DATA_OUTPUT": str(spec.data_output),
                    "REPLACE_DATA": "1" if options.replace_data else "0",
                },
            ),
        )

    for arm in selected_prepared:
        train_dependency = (
            f"afterok:{state.generate_job}" if state.generate_job else None
        )
        train_job = state.issue(
            "train",
            arm.name,
            _sbatch_command(
                repo_root,
                scripts / "train.sh",
                spec.resources["train"],
                job_name=f"v2-train-{arm.name}",
                log_dir=logs,
                dependency=train_dependency,
                exports={
                    "ARM": arm.name,
                    "MODEL_DIR": str(arm.model_dir),
                    "RESUME": "1" if options.resume else "0",
                    "TRAIN_CONFIG": str(arm.train_config),
                },
            ),
        )
        state.issue(
            "eval",
            arm.name,
            _sbatch_command(
                repo_root,
                scripts / "eval.sh",
                spec.resources["eval"],
                job_name=f"v2-eval-{arm.name}",
                log_dir=logs,
                dependency=f"afterok:{train_job}",
                exports={
                    "ARM": arm.name,
                    "EVAL_CONFIG": str(arm.eval_config),
                    "MODEL_DIR": str(arm.model_dir),
                    "RESULT_DIR": str(arm.result_dir),
                },
            ),
        )

    dependency_ids = state.submitted.copy()
    summary_dependency = (
        "afterany:" + ":".join(dependency_ids) if dependency_ids else "afterany:<jobs>"
    )
    state.issue(
        "summarize",
        None,
        _sbatch_command(
            repo_root,
            scripts / "summarize.sh",
            spec.resources["summarize"],
            job_name=f"v2-summary-{run_id}",
            log_dir=logs,
            dependency=summary_dependency,
            exports={"RUN_DIR": str(prepared.run_dir)},
        ),
    )
    return state.result()


def _validate_existing_views(arms: tuple[PreparedArm, ...], data_output: Path) -> None:
    missing = [
        path
        for arm in arms
        for path in (arm.train_view, arm.test_view)
        if not path.is_file()
    ]
    base = data_output.with_suffix("")
    manifest = base.with_name(base.name + "_manifest.json")
    if not manifest.is_file():
        missing.append(manifest)
    if missing:
        raise FileNotFoundError(
            "skip_generation requires existing validated views: "
            + ", ".join(str(path) for path in missing)
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("validation", {}).get("status") != "passed":
        raise ValueError(f"dataset manifest is not validated: {manifest}")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    dry_run: bool = typer.Option(
        False, help="Print the graph without sbatch or files."
    ),
    only: str | None = typer.Option(None, help="Comma-separated arm names."),
    skip_generation: bool = typer.Option(False),
    replace_data: bool = typer.Option(False),
    resume: bool = typer.Option(False),
    run_id: str | None = typer.Option(None),
) -> None:
    selected = (
        tuple(item.strip() for item in only.split(",") if item.strip())
        if only
        else None
    )
    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result = submit_experiment(
        EXPERIMENT_DIR / "run.yaml",
        REPO_ROOT,
        SubmissionOptions(
            run_id=identifier,
            dry_run=dry_run,
            only=selected,
            skip_generation=skip_generation,
            replace_data=replace_data,
            resume=resume,
        ),
    )
    if dry_run:
        for job in result.jobs:
            typer.echo(
                f"{job.stage}{':' + job.arm if job.arm else ''}: {shlex.join(job.command)}"
            )
    else:
        typer.echo(f"generate: {result.generate_job}")
        for arm, jobs in result.arms.items():
            typer.echo(f"{arm}: train={jobs['train']} eval={jobs['eval']}")
        typer.echo(f"summarize: {result.summary_job}")


if __name__ == "__main__":
    app()
