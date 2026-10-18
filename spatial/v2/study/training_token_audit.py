"""Audit persisted randomized-v2 training rows with the pinned model tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

_TOKENIZER = "Qwen/Qwen3.5-4B"
_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_THRESHOLDS = (4096, 8192, 12288, 16384, 24576, 32768)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: list[int], thresholds: tuple[int, ...]) -> dict[str, Any]:
    ordered = sorted(values)

    def percentile(value: float) -> int:
        return ordered[max(0, math.ceil(value * len(ordered)) - 1)]

    return {
        "min": ordered[0],
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "over_threshold": {
            str(threshold): sum(item > threshold for item in ordered)
            for threshold in thresholds
        },
    }


def audit_training_candidate(
    root: Path,
    *,
    tokenizer: Any,
    tokenizer_name: str,
    tokenizer_revision: str,
    thresholds: tuple[int, ...] = _THRESHOLDS,
) -> dict[str, Any]:
    """Verify candidate files and measure exact rendered chat-template lengths."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not thresholds or tuple(sorted(set(thresholds))) != thresholds:
        raise ValueError("thresholds must be unique and strictly increasing")
    file_results = {}
    global_training_max = 0
    for relative, receipt in sorted(manifest["files"].items()):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"training candidate file is missing: {relative}")
        if path.stat().st_size != receipt["bytes"] or _sha256(path) != receipt["sha256"]:
            raise ValueError(f"training candidate file changed: {relative}")
        training_tokens = []
        prompt_tokens = []
        target_tokens = []
        with path.open() as source:
            for line in source:
                row = json.loads(line)
                messages = row["messages"]
                training_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                prompt_text = tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                training_tokens.append(
                    len(tokenizer.encode(training_text, add_special_tokens=False))
                )
                prompt_tokens.append(
                    len(tokenizer.encode(prompt_text, add_special_tokens=False))
                )
                target_tokens.append(
                    len(
                        tokenizer.encode(
                            messages[-1]["content"], add_special_tokens=False
                        )
                    )
                )
        if len(training_tokens) != receipt["rows"]:
            raise ValueError(f"training candidate row count changed: {relative}")
        global_training_max = max(global_training_max, max(training_tokens))
        file_results[relative] = {
            "rows": len(training_tokens),
            "training_tokens": _distribution(training_tokens, thresholds),
            "evaluation_prompt_tokens": _distribution(prompt_tokens, thresholds),
            "target_tokens": _distribution(target_tokens, thresholds),
        }
    recommended = next(
        (threshold for threshold in thresholds if threshold >= global_training_max),
        None,
    )
    return {
        "schema": "spatialentail-randomized-v2-token-audit-v1",
        "status": "passed" if recommended is not None else "failed",
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "thresholds": list(thresholds),
        "recommended_train_max_tokens": recommended,
        "manifest_sha256": _sha256(manifest_path),
        "files": file_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", default=_TOKENIZER)
    parser.add_argument("--revision", default=_REVISION)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.revision,
    )
    audit = audit_training_candidate(
        args.candidate,
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer,
        tokenizer_revision=args.revision,
    )
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
