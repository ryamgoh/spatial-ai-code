"""Versioned SpatialMap-Corr entailment views; preserve historical releases."""

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import replace
from enum import Enum
from pathlib import Path

from spatial.v2.composition_contracts import UnsupportedEvidence, VerificationError
from spatial.v2.composition_execution import atomic_json
from spatial.v2.composition_feasibility import stable_int, token_measurement
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_training import (
    CONTEXT,
    MODEL,
    REVISION,
    STAGE1_CAP,
    STAGE2_CAP,
    completion_suffix,
    content_hash,
    digest,
    messages,
    prompt_text,
)
from spatial.v2.entailment_contracts import (
    VERSION,
    Contract,
    analyze,
    encode,
    parse_prompt,
    render_prompt,
)
from spatial.v2.entailment_dataset import menus_for
from spatial.v2.entailment_evidence import (
    REFERENCE_VERSION,
    build_evidence,
    reference_options,
    render_pair,
)
from spatial.v2.entailment_scene_protocol import (
    PLAIN_VERSIONS,
    SCOPE,
    SHORT_VERSION,
    all_scene_problem,
    display_references,
)
from spatial.v2.entailment_scene_protocol import (
    parse_prompt as parse_scene_prompt,
)
from spatial.v2.entailment_scene_protocol import (
    render_prompt as render_scene_prompt,
)
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import And, Direction, DirectionQuery, WhichQuery
from spatial.v2.text import SpatialTextAdapter

SCHEMA = "spatialmap-entailment-views-v1"


class SelectionScope(str, Enum):
    SOURCE = "source-options"
    ALL = "all-nonreference"


def adapt_problem(source_text, selection_scope, *, preserve_source_names=False):
    scope = SelectionScope(selection_scope)
    original = SpatialTextAdapter().parse(source_text).problem
    aliases = {name: f"Entity{i}" for i, name in enumerate(original.objects)}
    query = original.query
    if isinstance(query, DirectionQuery):
        query = replace(
            query,
            target=aliases[query.target],
            reference=aliases[query.reference],
            candidate_directions=frozenset(Direction),
        )
    else:
        candidates = (
            tuple(name for name in original.objects if name != query.reference)
            if isinstance(query, WhichQuery) and scope is SelectionScope.ALL
            else query.candidates
        )
        query = replace(
            query,
            reference=aliases[query.reference],
            candidates=tuple(aliases[name] for name in candidates),
        )
    problem = replace(
        original,
        objects=tuple(aliases[name] for name in original.objects),
        premise=And(
            tuple(
                replace(
                    atom,
                    subject=aliases[atom.subject],
                    reference=aliases[atom.reference],
                )
                for atom in original.premise.operands
            )
        ),
        query=query,
    )
    names = {alias: name for name, alias in aliases.items()}
    if preserve_source_names:
        native = {}
        paragraph = source_paragraph(source_text)
        directions = "|".join(d.value for d in Direction)
        for sentence in re.split(r"(?<=\.) ", paragraph):
            declaration = re.fullmatch(r"(.+?) is in the map\.", sentence)
            relation = re.fullmatch(
                rf"(.+?) is to the ({directions}) of (.+?)\.", sentence
            )
            entities = (
                (declaration.group(1),)
                if declaration
                else (relation.group(1), relation.group(3))
                if relation
                else ()
            )
            if not entities:
                raise ValueError("unsupported original scene sentence")
            for name in entities:
                normalized = name[4:] if name.lower().startswith("the ") else name
                if normalized in native and native[normalized] != name:
                    raise ValueError("source uses inconsistent original object names")
                native[normalized] = name
        if set(native) != set(aliases):
            raise ValueError("original source names differ from parsed scene objects")
        names = {alias: native[name] for name, alias in aliases.items()}
    return problem, names, original


def source_paragraph(source_text):
    """Preserve source sentences and order, normalizing only whitespace."""
    opening, separator, _ = source_text.partition("Please answer")
    if not separator:
        raise ValueError("external source lacks its question boundary")
    intro = "Consider a map with multiple objects:"
    if not opening.startswith(intro):
        raise ValueError("external source lacks its scene introduction")
    return " ".join(opening[len(intro) :].split())


def external_prompt(
    problem, menu, contract, names, *, prompt_version=None, scene_paragraph=None
):
    if prompt_version in PLAIN_VERSIONS:
        return render_scene_prompt(
            problem,
            menu,
            contract,
            display_names=names,
            scene_paragraph=scene_paragraph,
            prompt_version=prompt_version,
        )
    if prompt_version is not None:
        raise ValueError("unknown external prompt protocol")
    if set(names) != set(problem.objects) or len(set(names.values())) != len(names):
        raise ValueError("entity legend must be a bijection over all objects")
    lines = ["Entity aliases (same objects as the source scene):"]
    lines.extend(
        f"{alias} = {json.dumps(names[alias], ensure_ascii=True)}"
        for alias in sorted(names)
    )
    return "\n".join(lines) + "\n\n" + render_prompt(problem, menu, contract)


def parse_external_prompt(text, *, prompt_version=None, names=None):
    if prompt_version in PLAIN_VERSIONS:
        if names is None:
            raise ValueError("plain external prompts require reversible source names")
        return (
            *parse_scene_prompt(
                text, display_names=names, prompt_version=prompt_version
            ),
            names,
        )
    if prompt_version is not None:
        raise ValueError("unknown external prompt protocol")
    legend, body = text.split("\n\n", 1)
    lines = legend.splitlines()
    if lines[0] != "Entity aliases (same objects as the source scene):":
        raise ValueError("invalid external entity legend")
    names = {}
    for line in lines[1:]:
        alias, value = line.split(" = ", 1)
        if alias in names:
            raise ValueError("duplicate alias")
        names[alias] = json.loads(value)
        if not isinstance(names[alias], str):
            raise TypeError("entity name must be a string")
    problem, menu, contract = parse_prompt(body)
    if external_prompt(problem, menu, contract, names) != text:
        raise ValueError("external prompt does not match its complete template")
    return problem, menu, contract, names


def build_views(
    source, selection_scope, *, solver=None, omission_percent=20, prompt_version=None
):
    if prompt_version not in (None, *PLAIN_VERSIONS):
        raise ValueError("unknown external prompt protocol")
    plain = prompt_version in PLAIN_VERSIONS
    if plain and SelectionScope(selection_scope) is not SelectionScope.ALL:
        raise ValueError("plain external prompts require all-nonreference scope")
    solver = solver or QuestionSolver()
    problem, names, original = adapt_problem(
        source["text"],
        selection_scope,
        preserve_source_names=plain,
    )
    if plain:
        problem = all_scene_problem(problem)
    semantics = analyze(problem, solver=solver)
    categories = solver.negative_categories(problem)
    evidence, unsupported = None, None
    try:
        evidence = build_evidence(problem, solver=solver)
    except UnsupportedEvidence as exc:
        # External label evaluation permits absent reference prose. Solver errors
        # and failed verification are not converted into this exception.
        unsupported = str(exc)
    menus = menus_for(
        semantics, stable_int(source["id"]), omission_percent=omission_percent
    )
    story_id = source["id"].rsplit(".", 1)[0]
    rows = []
    for index, menu in enumerate(menus):
        for contract in (
            (Contract.UNIQUE,) if prompt_version == SHORT_VERSION else Contract
        ):
            decision = encode(semantics, menu, contract)
            prompt = external_prompt(
                problem,
                menu,
                contract,
                names,
                prompt_version=prompt_version,
                scene_paragraph=source_paragraph(source["text"]) if plain else None,
            )
            if parse_external_prompt(
                prompt, prompt_version=prompt_version, names=names
            ) != (problem, menu, contract, names):
                raise VerificationError(
                    "external source/domain/menu round trip changed"
                )
            references = (
                display_references(
                    render_pair(
                        evidence,
                        menu,
                        contract,
                        solver=solver,
                        restate_premises=plain,
                    ),
                    names if plain else None,
                )
                if evidence is not None
                else None
            )
            rows.append(
                {
                    "interface": VERSION,
                    "external_schema": SCHEMA,
                    "id": f"{source['id']}-m{index}-{contract.value}",
                    "base_id": source["id"],
                    "story_id": story_id,
                    "bundle_id": f"{source['id']}-m{index}",
                    "split": "external-test",
                    "query_kind": type(problem.query)
                    .__name__.removesuffix("Query")
                    .lower(),
                    "contract": contract.value,
                    "selection_scope": SelectionScope(selection_scope).value,
                    "menu": list(menu),
                    "menu_role": "primary" if index == 0 else "omission",
                    "answer": decision.letters,
                    "semantic_case": semantics.case,
                    "escape_reason": decision.escape_reason,
                    "entailed_count": len(semantics.entailed),
                    "entailed_menu_letters": "".join(
                        letter
                        for letter, value in zip("ABCD", menu)
                        if value in semantics.entailed
                    ),
                    "negative_menu_categories": {
                        letter: categories[value]
                        for letter, value in zip("ABCD", menu)
                        if value in categories
                    },
                    "coverage": "full"
                    if semantics.entailed <= set(menu)
                    else "partial",
                    "prompt": prompt,
                    "entity_names": names,
                    "source_text": source["text"],
                    "original_problem": tagged_dataclass_to_dict(original),
                    "problem": tagged_dataclass_to_dict(problem),
                    "references": references,
                    "reference_status": "available" if references else "unsupported",
                    "reference_reason": unsupported,
                }
            )
            if prompt_version is not None:
                rows[-1].update(
                    prompt_version=prompt_version,
                    scope_policy=SCOPE,
                    display_names=names,
                    reference_version=REFERENCE_VERSION,
                    scene_paragraph=source_paragraph(source["text"]),
                )
            if prompt_version == SHORT_VERSION:
                from spatial.v2.entailment_unique_study import STUDY_SCOPE

                rows[-1]["study_scope"] = STUDY_SCOPE
    return {
        "source_id": source["id"],
        "story_id": story_id,
        "scene_signature": content_hash(
            {
                "objects": list(original.objects),
                "premises": tagged_dataclass_to_dict(original.premise),
            }
        ),
        "semantics": tagged_dataclass_to_dict(semantics),
        "evidence": tagged_dataclass_to_dict(evidence)
        if evidence is not None
        else None,
        "rows": rows,
    }


def verify_view(row, *, solver=None):
    solver = solver or QuestionSolver()
    protocol = row.get("prompt_version")
    if (protocol, row.get("scope_policy")) not in (
        (None, None),
        *((version, SCOPE) for version in PLAIN_VERSIONS),
    ):
        raise VerificationError("unknown external scope/prompt protocol")
    plain = protocol in PLAIN_VERSIONS
    if protocol == SHORT_VERSION:
        from spatial.v2.entailment_unique_study import validate_row

        validate_row(row)
    if plain and row["selection_scope"] != SelectionScope.ALL.value:
        raise VerificationError("plain external view has restricted scope")
    problem, names, _ = adapt_problem(
        row["source_text"],
        row["selection_scope"],
        preserve_source_names=plain,
    )
    if plain:
        problem = all_scene_problem(problem)
        if row.get("display_names") != names:
            raise VerificationError("external display mapping changed")
    parsed, menu, contract, legend = parse_external_prompt(
        row["prompt"], prompt_version=protocol, names=names
    )
    if (
        parsed != problem
        or legend != names
        or row["entity_names"] != names
        or list(menu) != row["menu"]
        or contract.value != row["contract"]
    ):
        raise VerificationError(
            "external view changed source entities, query or domain"
        )
    if plain:
        scene = source_paragraph(row["source_text"])
        if row.get("scene_paragraph") != scene or row["prompt"] != external_prompt(
            problem,
            menu,
            contract,
            names,
            prompt_version=protocol,
            scene_paragraph=scene,
        ):
            raise VerificationError("external source paragraph changed")
    semantics = analyze(problem, solver=solver)
    decision = encode(semantics, menu, contract)
    if decision.letters != row["answer"]:
        raise VerificationError("external label differs from entailment computation")
    categories = solver.negative_categories(problem)
    letters = "".join(
        letter for letter, value in zip("ABCD", menu) if value in semantics.entailed
    )
    negatives = {
        letter: categories[value]
        for letter, value in zip("ABCD", menu)
        if value in categories
    }
    if (
        row["entailed_menu_letters"] != letters
        or row["negative_menu_categories"] != negatives
        or row["semantic_case"] != semantics.case
        or row["escape_reason"] != decision.escape_reason
    ):
        raise VerificationError("external semantic metadata differs from verification")
    if row["references"] is not None:
        evidence = build_evidence(problem, solver=solver)
        pair = display_references(
            render_pair(
                evidence, menu, contract, solver=solver, **reference_options(row)
            ),
            names if plain else None,
        )
        if pair != row["references"]:
            raise VerificationError(
                "external references differ from checked matched evidence"
            )


def verify_source_export(original_path, export_path, receipt_path):
    """Verify the projected Corr release, not its derived evaluation-wrapper fields."""
    receipt = json.loads(receipt_path.read_text())
    if digest(original_path) != receipt["input"]["sha256"]:
        raise ValueError(
            "original SpatialMap source hash differs from correction receipt"
        )
    originals = [
        json.loads(line)
        for line in original_path.read_text().splitlines()
        if line.strip()
    ]
    exported = [
        json.loads(line)
        for line in export_path.read_text().splitlines()
        if line.strip()
    ]
    expected = {row["id"]: row for row in originals}
    if (
        len(expected) != len(originals)
        or len(exported) != receipt["input"]["rows"]
        or len({r["id"] for r in exported}) != len(exported)
        or {r["id"] for r in exported} != set(expected)
    ):
        raise ValueError("external source IDs do not match the frozen correction")
    projection = [{key: row[key] for key in expected[row["id"]]} for row in exported]
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in projection
    ).encode()
    if hashlib.sha256(content).hexdigest() != receipt["outputs"]["corrected"]["sha256"]:
        raise ValueError(
            "projected corrected source hash differs from its release receipt"
        )
    return projection


def prepare(
    original,
    source,
    receipt,
    output,
    tokenizer,
    selection_scope,
    *,
    omission_percent=20,
    prompt_version=None,
):
    if output.exists() or not output.parent.is_dir():
        raise ValueError(
            "provide a fresh external candidate directory under an existing parent"
        )
    projection = verify_source_export(original, source, receipt)
    tokenizer_path = Path(tokenizer.name_or_path)
    if tokenizer_path.name != REVISION or not tokenizer_path.is_dir():
        raise ValueError("external preparation requires the pinned local tokenizer")
    paths = [
        original,
        source,
        receipt,
        *sorted(Path(__file__).parent.glob("*.py")),
        *(p for p in tokenizer_path.iterdir() if p.is_file()),
    ]
    hashes = {str(path): digest(path) for path in paths}
    results = []
    target_max = {"natural": 0, "symbolic": 0}
    reserved_max = 0
    solver = QuestionSolver()
    scenes = {}
    for item in projection:
        solver.clear()
        record = build_views(
            item,
            selection_scope,
            solver=solver,
            omission_percent=omission_percent,
            prompt_version=prompt_version,
        )
        previous = scenes.setdefault(record["story_id"], record["scene_signature"])
        if previous != record["scene_signature"]:
            raise VerificationError(
                "questions sharing a source story have different premises"
            )
        for row in record["rows"]:
            verify_view(row, solver=solver)
            for style in ("natural", "symbolic"):
                prefix = prompt_text(tokenizer, row, style)
                prompt_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
                reserved = (
                    prompt_tokens
                    + STAGE1_CAP
                    + STAGE2_CAP
                    + len(
                        tokenizer.encode(completion_suffix(), add_special_tokens=False)
                    )
                )
                if reserved > CONTEXT:
                    raise VerificationError(
                        "external input exceeds Stage-2 context reserve"
                    )
                reserved_max = max(reserved_max, reserved)
            if row["references"] is not None:
                for style in ("natural", "symbolic"):
                    measurement = token_measurement(tokenizer, row, style)
                    target_max[style] = max(
                        target_max[style], measurement["target_tokens"]
                    )
                    if not measurement["admitted"]:
                        raise VerificationError(
                            f"external token budget failed: {measurement}"
                        )
        results.append(record)
        if len(results) % 100 == 0:
            print(
                json.dumps(
                    {"external_bases_verified": len(results), "total": len(projection)}
                ),
                flush=True,
            )
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("external source or adapter changed during preparation")
    rows = [row for record in results for row in record["rows"]]
    output.mkdir()
    (output / "references.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (output / "evidence.jsonl").write_text(
        "".join(
            json.dumps({k: v for k, v in r.items() if k != "rows"}) + "\n"
            for r in results
        )
    )
    for style in ("natural", "symbolic"):
        (output / f"{style}-inputs.jsonl").write_text(
            "".join(
                json.dumps({"id": row["id"], "messages": messages(row, style)}) + "\n"
                for row in rows
            )
        )
    summary = {
        "schema": SCHEMA,
        "interface": VERSION,
        "prompt_version": prompt_version,
        "scope_policy": SCOPE if prompt_version in PLAIN_VERSIONS else None,
        "status": "unsealed-external-candidate-requires-review",
        "model": MODEL,
        "revision": REVISION,
        "context": CONTEXT,
        "selection_scope": SelectionScope(selection_scope).value,
        "paired_omission_percent": omission_percent,
        "base_questions": len(results),
        "source_stories": len(scenes),
        "rows_per_style": len(rows),
        "query_counts": dict(Counter(r["rows"][0]["query_kind"] for r in results)),
        "active_contracts": sorted({row["contract"] for row in rows}),
        "answer_distribution": dict(
            Counter(
                f"{row['query_kind']}/{row['contract']}/{row['answer']}" for row in rows
            )
        ),
        "reference_status": dict(Counter(row["reference_status"] for row in rows)),
        "max_target_tokens": target_max,
        "max_stage2_reserved_tokens": reserved_max,
        "source_hashes": hashes,
        "sealed": False,
        "training_authorized": False,
        "tokenizer_hashes": {
            p.name: digest(p) for p in tokenizer_path.iterdir() if p.is_file()
        },
        "output_hashes": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
    }
    if prompt_version == SHORT_VERSION:
        from spatial.v2.entailment_unique_study import STUDY_SCOPE

        summary["study_scope"] = STUDY_SCOPE
    atomic_json(output / "manifest.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-source", required=True, type=Path)
    parser.add_argument("--corr-export", required=True, type=Path)
    parser.add_argument("--correction-receipt", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--selection-scope", required=True, choices=[s.value for s in SelectionScope]
    )
    parser.add_argument("--omission-percent", type=int, default=20)
    parser.add_argument("--prompt-version", choices=list(PLAIN_VERSIONS))
    args = parser.parse_args()
    if args.tokenizer.name != REVISION or not args.tokenizer.is_dir():
        parser.error("provide the pinned local tokenizer")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = prepare(
        args.original_source,
        args.corr_export,
        args.correction_receipt,
        args.output,
        tokenizer,
        args.selection_scope,
        omission_percent=args.omission_percent,
        prompt_version=args.prompt_version,
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("source_hashes", "output_hashes")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
