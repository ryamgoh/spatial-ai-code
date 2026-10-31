"""Bounded correlated menu-position views; base semantics and family IDs stay fixed."""

from copy import deepcopy
from itertools import permutations

from spatial.v2.composition_feasibility import token_measurement
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.entailment_contracts import Contract, render_prompt
from spatial.v2.entailment_dataset import audit_row
from spatial.v2.entailment_evidence import reference_options, render_pair
from spatial.v2.entailment_production import annotate
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.entailment_scene_protocol import (
    VERSION as SCENE_VERSION,
)
from spatial.v2.entailment_scene_protocol import display_references
from spatial.v2.entailment_scene_protocol import render_prompt as render_scene_prompt

VERSION = "entailed-menu-position-views-v1"


def position_permutations(record):
    entailed = set(record["semantics"]["entailed"])
    if record["query_kind"] != "which" or len(entailed) not in (2, 3):
        return [(0, 1, 2, 3)]
    menu = next(row["menu"] for row in record["rows"] if row["menu_role"] == "primary")
    seen, result = set(), []
    for permutation in permutations(range(4)):
        letters = "".join(
            letter
            for letter, index in zip("ABCD", permutation)
            if menu[index] in entailed
        )
        if letters not in seen:
            seen.add(letters)
            result.append(permutation)
    return result


def permuted_letters(letters, permutation):
    if letters in ("E", "F"):
        return letters
    return "".join(
        letter
        for letter, old_index in zip("ABCD", permutation)
        if "ABCD"[old_index] in letters
    )


def balanced_metadata(record):
    result = deepcopy(record)
    rows = []
    for number, permutation in enumerate(position_permutations(record)):
        for old in record["rows"]:
            row = deepcopy(old)
            if number:
                row["id"] += f"-position{number}"
                row["bundle_id"] += f"-position{number}"
            row["position_variant"] = number
            row["menu"] = [old["menu"][i] for i in permutation]
            row["answer"] = permuted_letters(old["answer"], permutation)
            row["entailed_menu_letters"] = permuted_letters(
                old["entailed_menu_letters"], permutation
            )
            row["negative_menu_categories"] = {
                letter: old["negative_menu_categories"]["ABCD"[i]]
                for letter, i in zip("ABCD", permutation)
                if "ABCD"[i] in old["negative_menu_categories"]
            }
            rows.append(row)
    result["rows"] = rows
    result["menu_position_policy"] = VERSION
    annotate(rows)
    return result


def render_record(original, metadata, tokenizer):
    evidence = load_evidence(original)
    solver = QuestionSolver()
    result = deepcopy(original)
    result["rows"], result["tokens"] = [], []
    for row in metadata["rows"]:
        row = deepcopy(row)
        menu, contract = tuple(row["menu"]), Contract(row["contract"])
        renderer = (
            render_scene_prompt
            if row.get("prompt_version") == SCENE_VERSION
            else render_prompt
        )
        kwargs = (
            {
                "display_names": row.get("display_names"),
                "scene_paragraph": row.get("scene_paragraph"),
                "layout": "paragraph"
                if row["prompt"].splitlines()[2] == ""
                else "lines",
            }
            if renderer is render_scene_prompt
            else {}
        )
        row["prompt"] = renderer(evidence.semantics.problem, menu, contract, **kwargs)
        row["references"] = display_references(
            render_pair(
                evidence, menu, contract, solver=solver, **reference_options(row)
            ),
            row.get("display_names"),
        )
        audit_row(row, evidence, solver=solver)
        for style in row["references"]:
            tokens = token_measurement(tokenizer, row, style)
            if not tokens["admitted"]:
                raise ValueError(
                    f"revised reference exceeds the token budget: {tokens}"
                )
            result["tokens"].append(tokens)
        result["rows"].append(row)
    result["menu_position_policy"] = VERSION
    return result
