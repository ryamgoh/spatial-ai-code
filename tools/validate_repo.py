#!/usr/bin/env python3
"""Fast, GPU-free checks for experiment configs and Slurm launchers."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # Keep shell/layout validation usable before uv sync.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
REQUIRED_SBATCH = {
    "job-name",
    "partition",
    "nodes",
    "ntasks",
    "cpus-per-task",
    "mem",
    "time",
    "gres",
    "output",
    "error",
}


if yaml is not None:

    class TaggedSafeLoader(yaml.SafeLoader):
        """Load local task tags such as !function without importing their code."""


    def _unknown_tag(loader: TaggedSafeLoader, node: yaml.Node) -> object:
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        return loader.construct_mapping(node)


    TaggedSafeLoader.add_constructor(None, _unknown_tag)


def parse_duration(value: str) -> int:
    """Return a Slurm D-HH:MM:SS or HH:MM:SS duration in seconds."""
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", value)
    if not match:
        raise ValueError(f"unsupported duration {value!r}")
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def load_yaml_files(errors: list[str]) -> int:
    paths = sorted(EXPERIMENTS.rglob("*.yaml"))
    if yaml is None:
        ruby = shutil.which("ruby")
        if ruby is None:
            return 0
        result = subprocess.run(
            [ruby, "-ryaml", "-e", "ARGV.each { |path| YAML.parse_file(path) }", *map(str, paths)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            errors.append(f"invalid YAML: {result.stderr.strip()}")
        return len(paths)
    count = 0
    for path in paths:
        count += 1
        try:
            with path.open(encoding="utf-8") as stream:
                yaml.load(stream, Loader=TaggedSafeLoader)
        except Exception as exc:
            errors.append(f"{path.relative_to(ROOT)}: invalid YAML: {exc}")
    return count


def sbatch_directives(text: str) -> dict[str, str]:
    directives: dict[str, str] = {}
    for name, value in re.findall(r"^#SBATCH\s+--([\w-]+)(?:=|\s+)(\S+)", text, re.M):
        directives[name] = value
    return directives


def check_shell_files(errors: list[str]) -> int:
    paths = sorted((ROOT / "slurm").rglob("*.sh"))
    paths += sorted(EXPERIMENTS.glob("**/slurm/*.sh"))

    for path in paths:
        relative = path.relative_to(ROOT)
        result = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, check=False
        )
        if result.returncode:
            errors.append(f"{relative}: bash -n failed: {result.stderr.strip()}")

        text = path.read_text(encoding="utf-8")
        directives = sbatch_directives(text)
        is_helper = path.is_relative_to(ROOT / "slurm" / "lib")
        is_driver = path.name.endswith("-driver.sh") and not directives
        if is_helper:
            continue
        if is_driver:
            continue

        missing = sorted(REQUIRED_SBATCH - directives.keys())
        if missing:
            errors.append(f"{relative}: missing #SBATCH directives: {', '.join(missing)}")
            continue

        if path.is_relative_to(EXPERIMENTS):
            experiment_dir = path.parent.parent
            expected_prefix = f"{experiment_dir.relative_to(ROOT)}/logs/%x-%j."
        else:
            experiment_dir = ROOT / "slurm" / "archive"
            expected_prefix = "slurm/archive/logs/%x-%j."
        expected_logs = {
            "output": f"{expected_prefix}out",
            "error": f"{expected_prefix}err",
        }
        for directive, expected in expected_logs.items():
            if directives[directive] != expected:
                errors.append(
                    f"{relative}: --{directive} must be {expected}"
                )
        if not (experiment_dir / "logs" / ".gitkeep").is_file():
            log_marker = experiment_dir.relative_to(ROOT) / "logs" / ".gitkeep"
            errors.append(f"{relative}: missing tracked {log_marker}")

        if directives["partition"] == "gpu":
            try:
                if parse_duration(directives["time"]) > 3 * 3600:
                    errors.append(f"{relative}: gpu partition exceeds its 03:00:00 limit")
            except ValueError as exc:
                errors.append(f"{relative}: {exc}")

        if "srun " in text and "pin-srun-cpus.sh" not in text:
            errors.append(f"{relative}: invokes srun without the shared CPU-pin helper")

        source_pattern = r'^source\s+"?\$\{SLURM_SUBMIT_DIR[^}]*\}/((?:experiments|slurm)/[^\s"\']+\.sh)'
        for source in re.findall(source_pattern, text, re.M):
            if not (ROOT / source).is_file():
                errors.append(f"{relative}: sourced file does not exist: {source}")

    return len(paths)


def check_layout(errors: list[str]) -> None:
    root_launchers = sorted(ROOT.glob("run_*.sh"))
    if root_launchers:
        names = ", ".join(path.name for path in root_launchers)
        errors.append(f"root launchers must live with their experiment: {names}")


def check_documented_launchers(errors: list[str]) -> None:
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    paths += sorted(EXPERIMENTS.glob("**/README.md"))
    paths += sorted(EXPERIMENTS.glob("**/RESULTS.md"))
    pattern = r"sbatch(?:\s+--[^\s]+)*\s+(experiments/[^\s`]+\.sh)"
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(pattern, text):
            if not (ROOT / target).is_file():
                errors.append(
                    f"{path.relative_to(ROOT)}: documented launcher does not exist: {target}"
                )


def main() -> int:
    errors: list[str] = []
    yaml_count = load_yaml_files(errors)
    shell_count = check_shell_files(errors)
    check_layout(errors)
    check_documented_launchers(errors)

    if errors:
        print(f"Repository validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    yaml_status = f"{yaml_count} YAML files" if yaml_count else "YAML skipped (install PyYAML or Ruby)"
    print(f"Repository validation passed: {yaml_status}, {shell_count} shell files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
