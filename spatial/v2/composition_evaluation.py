"""Two-stage composition evaluation and dev-only epoch selection.

vLLM is imported only by the GPU runner. Pure scoring/replay helpers are tested
locally; no model loading or generation happens during dataset preparation.
"""

import argparse
import json
from collections import Counter
from fractions import Fraction
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

from spatial.v2.composition_contracts import parse_answer
from spatial.v2.composition_training import (
    CONTEXT,
    MODEL,
    REVISION,
    STAGE1_CAP,
    STAGE2_CAP,
    completion_suffix,
    content_hash,
    digest,
    prompt_text,
    verify_prepared,
    write_jsonl,
)
from spatial.v2.entailment_contracts import (
    VERSION as ENTAILMENT_VERSION,
)
from spatial.v2.entailment_contracts import (
    Contract,
)
from spatial.v2.entailment_contracts import (
    parse_answer as parse_entailment_answer,
)


def allowed_answers(contract):
    if contract in {"single", Contract.UNIQUE.value}:
        return list("ABCDEF")
    if contract not in {
        "visible-possible",
        "all-possible",
        Contract.LISTED.value,
        Contract.EXHAUSTIVE.value,
    }:
        raise ValueError("unknown contract")
    return ["".join(c) for n in range(1, 5) for c in combinations("ABCD", n)] + [
        "E",
        "F",
    ]


def parse_or_none(text, contract, stage):
    try:
        parser = (
            parse_entailment_answer
            if contract in {c.value for c in Contract}
            else parse_answer
        )
        return parser(text, contract, stage=stage)
    except ValueError:
        return None


def replay_tokens(prompt_ids, response_ids, suffix_ids, context=CONTEXT):
    result = [*prompt_ids, *response_ids, *suffix_ids]
    if len(result) + STAGE2_CAP > context:
        raise ValueError("full Stage-1 replay exceeds context; truncation is forbidden")
    return result


def adapter_hash(path):
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    weights = [p for p in files if p.name.startswith("adapter_model.")]
    if not (path / "adapter_config.json").is_file() or not weights:
        raise ValueError("adapter config and weights are required")
    return content_hash(
        {
            str(p.relative_to(path)): digest(p)
            for p in files
            if p.name == "adapter_config.json" or p in weights
        }
    )


def evaluation_contracts(rows):
    from spatial.v2.entailment_unique_study import STUDY_SCOPE, validate_row

    contracts = {row["contract"] for row in rows}
    old_contracts = {"single", "visible-possible", "all-possible"}
    new_contracts = {c.value for c in Contract}
    scopes = {row.get("study_scope") for row in rows}
    if scopes != {None}:
        if scopes != {STUDY_SCOPE} or not rows:
            raise ValueError("evaluation cannot mix or guess study scopes")
        if len({r.get("prompt_version") for r in rows}) != 1:
            raise ValueError("evaluation cannot mix prompt versions")
        for row in rows:
            validate_row(row)
        return (Contract.UNIQUE.value,)
    if contracts == new_contracts:
        if any(row.get("interface") != ENTAILMENT_VERSION for row in rows):
            raise ValueError("entailment evaluation requires its versioned interface")
        from spatial.v2.entailment_scene_protocol import SCOPE, VERSION

        protocols = {(r.get("prompt_version"), r.get("scope_policy")) for r in rows}
        if len(protocols) != 1:
            raise ValueError("evaluation cannot mix scoped and all-scene protocols")
        if not protocols <= {(None, None), (VERSION, SCOPE)}:
            raise ValueError("unknown evaluation prompt protocol")
        return tuple(c.value for c in Contract)
    elif contracts == old_contracts:
        if any(row.get("interface") == ENTAILMENT_VERSION for row in rows):
            raise ValueError("entailment interface cannot use possibility contracts")
        return ("single", "visible-possible", "all-possible")
    else:
        raise ValueError("all three contracts from one semantic version are required")


def score(rows, outputs):
    expected = {r["id"]: r for r in rows}
    found = {r["id"]: r for r in outputs}
    if (
        len(expected) != len(rows)
        or len(found) != len(outputs)
        or set(expected) != set(found)
    ):
        raise ValueError("evaluation IDs must match the frozen split exactly")
    contract_order = evaluation_contracts(rows)
    answers = {
        stage: {
            key: parse_or_none(
                found[key][f"stage{stage}"]["text"], row["contract"], stage
            )
            for key, row in expected.items()
        }
        for stage in (1, 2)
    }
    totals = {}
    for stage in (1, 2):
        correct = {
            key: answers[stage][key] == row["answer"] for key, row in expected.items()
        }
        by_contract = {}
        for contract in contract_order:
            keys = [k for k, row in expected.items() if row["contract"] == contract]
            if not keys:
                raise ValueError("every contract must be present")
            by_contract[contract] = {
                "correct": sum(correct[k] for k in keys),
                "total": len(keys),
            }
        macro = sum(
            Fraction(c["correct"], c["total"]) for c in by_contract.values()
        ) / len(contract_order)
        totals[f"stage{stage}"] = {
            "correct": sum(correct.values()),
            "total": len(correct),
            "contracts": by_contract,
            "macro_accuracy": float(macro),
            "format_valid": sum(
                answer is not None for answer in answers[stage].values()
            ),
            "finish_reasons": dict(
                Counter(
                    r[f"stage{stage}"].get("finish_reason", "unrecorded")
                    for r in outputs
                )
            ),
            "slices": {},
        }
        for field in (
            "query_kind",
            "semantic_case",
            "escape_reason",
            "coverage",
            "recipe",
            "gold_answer_class",
            "depth",
            "entailed_count",
        ):
            groups = {}
            for key, row in expected.items():
                if field == "gold_answer_class":
                    label = (
                        row["answer"] if row["answer"] in {"E", "F"} else "substantive"
                    )
                elif field not in row:
                    continue
                else:
                    label = str(row[field])
                cell = groups.setdefault(label, {"correct": 0, "total": 0})
                cell["correct"] += correct[key]
                cell["total"] += 1
            totals[f"stage{stage}"]["slices"][field] = groups
        if Contract.LISTED.value in contract_order:
            option_cells = {
                label: {"correct": 0, "total": 0}
                for label in ("entailed", "contingent", "impossible")
            }
            for key, row in expected.items():
                # Listed directly labels displayed claims. Unique/Exhaustive
                # escapes do not expose a full claim classification.
                if (
                    row["contract"] != Contract.LISTED.value
                    or row.get("semantic_case") == "inconsistent"
                ):
                    continue
                predicted = answers[stage][key]
                interpretable = predicted is not None and predicted != "E"
                for letter in "ABCD":
                    entailed = letter in row["entailed_menu_letters"]
                    label = (
                        "entailed"
                        if entailed
                        else row["negative_menu_categories"][letter]
                    )
                    option_cells[label]["total"] += 1
                    option_cells[label]["correct"] += interpretable and (
                        (letter in predicted) == entailed
                    )
            totals[f"stage{stage}"]["listed_option_classification"] = option_cells
    totals["generated_tokens"] = {
        f"stage{stage}": sum(len(r[f"stage{stage}"]["token_ids"]) for r in outputs)
        for stage in (1, 2)
    }
    totals["input_tokens"] = {
        f"stage{stage}": sum(r[f"stage{stage}"]["input_tokens"] for r in outputs)
        for stage in (1, 2)
    }
    transitions = {}
    for key, row in expected.items():
        first, second = answers[1][key], answers[2][key]
        transitions[key] = {
            "answer_changes": first != second,
            "rescues": first != row["answer"] and second == row["answer"],
            "regressions": first == row["answer"] and second != row["answer"],
        }
    for metric in ("rescues", "regressions"):
        totals[metric] = sum(t[metric] for t in transitions.values())
    totals["transitions"] = {}
    for status in ("length", "completed"):
        selected = [
            r
            for r in rows
            if (found[r["id"]]["stage1"].get("finish_reason") == "length")
            == (status == "length")
        ]
        totals["transitions"][status] = {
            "total": len(selected),
            **{
                metric: sum(transitions[row["id"]][metric] for row in selected)
                for metric in ("answer_changes", "rescues", "regressions")
            },
        }
    totals["generated_tokens"]["total"] = sum(
        totals["generated_tokens"][f"stage{s}"] for s in (1, 2)
    )
    return totals


def select_epoch(rows, runs, style, manifest_hash):
    if len(runs) != 2 or {r["epoch"] for r in runs} != {1, 2}:
        raise ValueError("both frozen epoch evaluations are required")
    ranked = []
    for run in runs:
        if (
            run["split"] != "dev"
            or run["style"] != style
            or run["manifest_sha256"] != manifest_hash
            or run["model"] != MODEL
            or run["revision"] != REVISION
            or not run.get("adapter")
            or not run.get("adapter_sha256")
        ):
            raise ValueError("checkpoint selection requires matching dev provenance")
        scores = score(rows, run["outputs"])
        cells = scores["stage1"]["contracts"]
        macro = sum(Fraction(c["correct"], c["total"]) for c in cells.values()) / len(
            cells
        )
        ranked.append((macro, -run["epoch"], run))
    selected = max(ranked, key=lambda item: item[:2])[2]
    return {
        "epoch": selected["epoch"],
        "adapter": selected["adapter"],
        "adapter_sha256": selected["adapter_sha256"],
        "manifest_sha256": manifest_hash,
        "style": style,
        "rule": (
            "Unique Stage-1 dev view accuracy; earliest epoch breaks ties"
            if evaluation_contracts(rows) == (Contract.UNIQUE.value,)
            else "Stage-1 macro-contract dev accuracy; earliest epoch breaks ties"
        ),
    }


def verify_seal(prepared):
    prepared = Path(prepared)
    verify_prepared(prepared)
    seal = json.loads((prepared / "seal.json").read_text())
    if seal["status"] != "frozen-prepared-not-launched":
        raise ValueError("package is not frozen")
    if seal["manifest_sha256"] != digest(prepared / "manifest.json"):
        raise ValueError("seal no longer matches manifest")
    if seal["audit_sha256"] != digest(prepared / "audit-review.json"):
        raise ValueError("audit review changed")
    return seal


def run_gpu(prepared, output, style, split, adapter=None, epoch=None, smoke=False):
    prepared, output = Path(prepared), Path(output)
    verify_seal(prepared)
    if output.exists():
        raise ValueError("refusing to overwrite an evaluation result")
    if output.with_suffix(".partial.jsonl").exists():
        raise ValueError("partial result exists; use a new output path")
    if smoke and split != "dev":
        raise ValueError("smoke checks must use dev, never test")
    if adapter is not None and split == "dev" and epoch not in {1, 2}:
        raise ValueError("dev adapters require their epoch number")
    if adapter is not None and split == "test":
        selection = json.loads((prepared / f"selected-{style}.json").read_text())
        if (
            selection["adapter"] != str(adapter)
            or selection["manifest_sha256"] != digest(prepared / "manifest.json")
            or selection["style"] != style
            or selection["adapter_sha256"] != adapter_hash(adapter)
        ):
            raise ValueError("test adapter differs from the dev-selected checkpoint")
    frozen_adapter_hash = adapter_hash(adapter) if adapter else None
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    manifest = json.loads((prepared / "manifest.json").read_text())
    snapshot = Path(
        snapshot_download(
            MODEL,
            revision=REVISION,
            allow_patterns=["tokenizer*", "chat_template.jinja", "config.json"],
        )
    )
    for name, expected in manifest["tokenizer_sha256"].items():
        if digest(snapshot / name) != expected:
            raise ValueError(f"runtime tokenizer changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    rows = [r for r in read_jsonl(prepared / "references.jsonl") if r["split"] == split]
    contract_order = evaluation_contracts(rows)
    if smoke:
        if contract_order == (Contract.UNIQUE.value,):
            rows = [
                next(r for r in rows if r["query_kind"] == kind)
                for kind in ("direction", "which", "count")
            ]
        else:
            rows = [
                next(r for r in rows if r["contract"] == contract)
                for contract in contract_order
            ]
    engine = LLM(
        model=MODEL,
        revision=REVISION,
        tokenizer_revision=REVISION,
        max_model_len=CONTEXT,
        enable_lora=adapter is not None,
        max_lora_rank=64,
        seed=42,
        trust_remote_code=False,
    )
    request = LoRARequest(style, 1, str(adapter)) if adapter else None
    records = []
    stop_ids = {
        tokenizer.convert_tokens_to_ids(token)
        for token in ("<|im_end|>", "<|endoftext|>")
    }
    for row in rows:
        ids = tokenizer.encode(
            prompt_text(tokenizer, row, style), add_special_tokens=False
        )
        if len(ids) + STAGE1_CAP > CONTEXT:
            raise ValueError("Stage-1 context overflow")
        first = engine.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=STAGE1_CAP,
                stop=["<|im_end|>", "<|endoftext|>"],
            ),
            lora_request=request,
            use_tqdm=False,
        )[0].outputs[0]
        terminated = bool(first.token_ids) and first.token_ids[-1] in stop_ids
        suffix_ids = tokenizer.encode(
            completion_suffix(terminated=terminated), add_special_tokens=False
        )
        second_ids = replay_tokens(ids, first.token_ids, suffix_ids)
        second = engine.generate(
            [{"prompt_token_ids": second_ids}],
            SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=STAGE2_CAP,
                structured_outputs=StructuredOutputsParams(
                    choice=allowed_answers(row["contract"])
                ),
                stop=["<|im_end|>", "<|endoftext|>"],
            ),
            lora_request=request,
            use_tqdm=False,
        )[0].outputs[0]
        records.append(
            {
                "id": row["id"],
                "stage1": {
                    "text": first.text,
                    "token_ids": list(first.token_ids),
                    "finish_reason": first.finish_reason,
                    "input_tokens": len(ids),
                },
                "stage2": {
                    "text": second.text,
                    "token_ids": list(second.token_ids),
                    "finish_reason": second.finish_reason,
                    "input_tokens": len(second_ids),
                },
            }
        )
        write_jsonl(output.with_suffix(".partial.jsonl"), records)
    result = {
        "split": "smoke" if smoke else split,
        "style": style,
        "model": MODEL,
        "revision": REVISION,
        "adapter": str(adapter) if adapter else None,
        "adapter_sha256": frozen_adapter_hash,
        "epoch": epoch,
        "manifest_sha256": digest(prepared / "manifest.json"),
        "outputs": records,
        "scores": score(rows, records),
        "runtime_versions": {
            name: version(name)
            for name in ("vllm", "torch", "transformers", "tokenizers")
        },
    }
    if adapter and adapter_hash(adapter) != frozen_adapter_hash:
        raise ValueError("adapter changed during evaluation")
    output.write_text(json.dumps(result, indent=2) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--style", choices=("natural", "symbolic"), required=True)
    run.add_argument("--split", choices=("dev", "test"), required=True)
    run.add_argument("--adapter", type=Path)
    run.add_argument("--epoch", type=int)
    run.add_argument("--smoke", action="store_true")
    choose = sub.add_parser("select")
    choose.add_argument("--prepared", type=Path, required=True)
    choose.add_argument("--style", choices=("natural", "symbolic"), required=True)
    choose.add_argument("runs", type=Path, nargs=2)
    args = parser.parse_args()
    if args.command == "run":
        run_gpu(
            args.prepared,
            args.output,
            args.style,
            args.split,
            args.adapter,
            args.epoch,
            args.smoke,
        )
    else:
        verify_seal(args.prepared)
        selected_path = args.prepared / f"selected-{args.style}.json"
        if selected_path.exists():
            raise ValueError("refusing to overwrite checkpoint selection")
        rows = [
            r
            for r in read_jsonl(args.prepared / "references.jsonl")
            if r["split"] == "dev"
        ]
        result = select_epoch(
            rows,
            [json.loads(p.read_text()) for p in args.runs],
            args.style,
            digest(args.prepared / "manifest.json"),
        )
        result["dev_runs_sha256"] = {str(p): digest(p) for p in args.runs}
        selected_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
