from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_launchers_request_only_declared_project_extras() -> None:
    project = tomllib.loads((REPO / "pyproject.toml").read_text())
    declared = set(project.get("project", {}).get("optional-dependencies", {}))
    offenders = []
    for path in (REPO / "experiments").rglob("*.sh"):
        text = path.read_text()
        for extra in re.findall(r"uv\s+sync\s+--extra\s+(\S+)", text):
            if extra not in declared:
                offenders.append(f"{path.relative_to(REPO)}: {extra}")

    assert not offenders, "launchers request undeclared uv extras: " + ", ".join(
        offenders
    )
