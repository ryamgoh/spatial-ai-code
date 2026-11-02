"""Shared scoring, adapter identity, and answer/replay contract helpers.

Current inference and dev selection live in entailment_evaluation.
"""

from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from spatial.v2.composition_contracts import parse_answer
from spatial.v2.composition_training import (
    CONTEXT,
    STAGE2_CAP,
    content_hash,
    digest,
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
