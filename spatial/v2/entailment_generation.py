"""Run-local candidate generation with explicit experiment policy and paths."""

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from spatial.v2.composition_balance import query_profile
from spatial.v2.composition_pilot import story_key
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_training import messages
from spatial.v2.entailment_artifact_audit import (
    audit_record,
    checked_sat,
    independent_claims,
    independent_plan_check,
    label,
    replay,
    require,
)
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_dataset import (
    audit_row,
    build_attempt,
    selected_explanation_depth,
)
from spatial.v2.entailment_evidence import SPATIAL_PROSE_VERSION, render_pair
from spatial.v2.entailment_external import (
    SelectionScope,
    adapt_problem,
    build_views,
    external_prompt,
    verify_source_export,
    verify_view,
)
from spatial.v2.entailment_generator import SceneRequest, build_scene
from spatial.v2.entailment_readable_v3 import make_plan
from spatial.v2.entailment_retention import matches
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.entailment_scene_protocol import SHORT_VERSION, parse_row, render_prompt
from spatial.v2.entailment_thinking_training import tokenize
from spatial.v2.serialization import atomic_json
from spatial.v2.solver import CountQuery, Direction, DirectionQuery
from spatial.v2.text import SpatialTextAdapter

DEPENDENCIES = ("transformers", "tokenizers", "jinja2", "pyyaml", "z3-solver", "scipy")


@dataclass(frozen=True)
class GenerationConfig:
    """Experiment inputs; no shared module imports an experiment to obtain them."""

    root: Path
    experiment: Path
    policy: dict
    slot_type: type
    smoke_slots: tuple
    coverage_policy: Path
    original_source: Path
    corr_source: Path
    correction_receipt: Path
    pending_release_checks: tuple[str, ...]
    anchor_prefix: str
    family_prefix: str
    manifest_schema: str


_WORKER = None


def initialize_worker(config, admission_path, active_path):
    global _WORKER
    _WORKER = GenerationWorkflow(config)
    _WORKER.initialize(admission_path, active_path)


def generate_worker(job):
    require(_WORKER is not None, "generation worker was not initialized")
    return _WORKER.generate_one(job)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def append_json(stream, value):
    stream.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
    stream.flush()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def identities(problem):
    return {
        "query_key": query_profile(problem)["key"],
        "scene_key": story_key(problem.objects, problem.premise),
    }


class Inventory:
    def __init__(self):
        self.queries = set()
        self.scenes = {}
        self.families = {}

    def conflict(self, keys, split, family=None):
        if keys["query_key"] in self.queries:
            return "duplicate-query"
        if keys["scene_key"] in self.scenes and self.scenes[keys["scene_key"]] != split:
            return "cross-split-scene"
        if family in self.families and self.families[family] != split:
            return "cross-split-family"
        return None

    def add(self, keys, split, family=None, *, allow_external_duplicate=False):
        reason = self.conflict(keys, split, family)
        require(
            reason is None
            or (allow_external_duplicate and reason == "duplicate-query"),
            f"inventory conflict: {reason}",
        )
        self.queries.add(keys["query_key"])
        self.scenes[keys["scene_key"]] = split
        if family is not None:
            self.families[family] = split


def rebind_menu(row, evidence, menu, *, solver, external=False):
    """Recompute all menu-dependent fields; never edit only the answer letter."""
    from spatial.v2.entailment_contracts import encode

    semantics = (
        evidence.semantics
        if evidence is not None
        else solver.analyze_entailment(parse_row(row)[0])
    )
    decision = encode(semantics, menu, Contract.UNIQUE)
    categories = solver.negative_categories(semantics.problem)
    row.update(
        menu=list(menu),
        answer=decision.letters,
        escape_reason=decision.escape_reason,
        entailed_menu_letters="".join(
            c for (c, v) in zip("ABCD", menu) if v in semantics.entailed
        ),
        displayed_entailed_count=len(set(menu) & semantics.entailed),
        negative_menu_categories={
            c: categories[v] for (c, v) in zip("ABCD", menu) if v in categories
        },
        has_contingent_negative=any(categories.get(v) == "contingent" for v in menu),
        has_impossible_negative=any(categories.get(v) == "impossible" for v in menu),
        coverage="full" if semantics.entailed <= set(menu) else "partial",
    )
    if external:
        row["prompt"] = external_prompt(
            semantics.problem,
            menu,
            Contract.UNIQUE,
            row["entity_names"],
            prompt_version=SHORT_VERSION,
            scene_paragraph=row["scene_paragraph"],
        )
    else:
        row["prompt"] = render_prompt(
            semantics.problem, menu, Contract.UNIQUE, prompt_version=SHORT_VERSION
        )
    if evidence is not None:
        from spatial.v2.entailment_scene_protocol import display_references

        row["references"] = display_references(
            render_pair(
                evidence,
                menu,
                Contract.UNIQUE,
                solver=solver,
                restate_premises=True,
                spatial_prose=True,
            ),
            row.get("display_names"),
        )
        row["reference_version"] = SPATIAL_PROSE_VERSION
    if not external:
        depth = selected_explanation_depth(evidence, menu, Contract.UNIQUE)
        row["selected_explanation_depth"] = depth
        row["selected_explanation_depth_support"] = (
            depth is not None and depth == row["depth"]
        )
        row["substantive_depth_support"] = (
            row["answer"] in "ABCD" and row["selected_explanation_depth_support"]
        )
        row["decision_depth_support"] = (
            row["selected_explanation_depth_support"]
            if decision.escape_reason == "multiple-entailed"
            else row["substantive_depth_support"]
        )


def open_split(stack, run, split):
    directory = run / "data" / split
    directory.mkdir(parents=True, exist_ok=False)
    names = [
        "references",
        "evidence",
        "token-measurements",
        "natural-inputs",
        "symbolic-inputs",
    ]
    if split in ("train", "dev"):
        names += [
            "natural-chat",
            "symbolic-chat",
            "natural-tokenized",
            "symbolic-tokenized",
            "index",
        ]
    return {
        name: stack.enter_context((directory / f"{name}.jsonl").open("x"))
        for name in names
    }


def write_synthetic(handles, result):
    record = result["record"]
    row = record["rows"][0]
    append_json(handles["evidence"], record)
    append_json(handles["references"], row)
    for measurement in record["tokens"]:
        append_json(handles["token-measurements"], measurement)
    index = {"id": row["id"], "base_id": row["base_id"], "token_sha256": {}}
    for style in ("natural", "symbolic"):
        append_json(
            handles[f"{style}-inputs"],
            {"id": row["id"], "messages": messages(row, style)},
        )
        if style in result["tokenized"]:
            tokens = result["tokenized"][style]
            append_json(handles[f"{style}-tokenized"], tokens)
            append_json(
                handles[f"{style}-chat"],
                {"id": row["id"], "messages": result["chats"][style]},
            )
            index["token_sha256"][style] = hashlib.sha256(
                json.dumps(tokens, sort_keys=True).encode()
            ).hexdigest()
    if "index" in handles:
        append_json(handles["index"], index)


def payload_hashes(run):
    return {
        p.relative_to(run).as_posix(): digest(p)
        for directory in ("inputs", "data", "tokenizers", "audit")
        for p in sorted((run / directory).rglob("*"))
        if p.is_file() and ".cache" not in p.parts
    }


def select_review_records(records):
    """Deterministic review coverage; selection is not a completed human review."""
    ordered = sorted(
        records,
        key=lambda r: hashlib.sha256(
            f"review-selection-v2/{r['rows'][0]['id']}".encode()
        ).hexdigest(),
    )
    groups = defaultdict(list)
    longest = {}
    for record in ordered:
        split, kind, level = (
            record["split"],
            record["query_kind"],
            record["construction_level"],
        )
        groups[f"case/{kind}/{record['case']}"].append(record)
        groups[f"split-level/{split}/{kind}/{level}"].append(record)
        if record["slot"]["contingent"]:
            groups[f"contingent/{split}/{level}"].append(record)
        if kind == "count" and record["case"] in ("A-D", "F"):
            value = record["slot"]["count_value"]
            groups[f"count-value/{split}/{level}/{value}"].append(record)
        for token in record["tokens"]:
            key = f"longest/{split}/{token['style']}"
            length = token["training_tokens"]
            if key not in longest or length > longest[key][0]:
                longest[key] = (length, record)
    selected, reasons = {}, defaultdict(list)

    def include(record, reason):
        identity = record["rows"][0]["id"]
        selected[identity] = record
        reasons[identity].append(reason)

    for key, members in sorted(groups.items()):
        for record in members[: 10 if key.startswith("case/") else 1]:
            include(record, key)
    for key, (_, record) in sorted(longest.items()):
        include(record, key)
    ids = sorted(selected)
    return [selected[i] for i in ids], {
        "version": "review-selection-v2",
        "status": "selected-not-reviewed",
        "selected_count": len(ids),
        "selected": [{"id": i, "reasons": reasons[i]} for i in ids],
        "available_case_counts": {
            key: len(members)
            for key, members in groups.items()
            if key.startswith("case/")
        },
    }


class GenerationWorkflow:
    def __init__(self, config):
        self.config = config
        self.tokenizers = None

    def fingerprints(self):
        paths = [
            *sorted((self.config.root / "spatial/v2").glob("*.py")),
            *sorted(self.config.experiment.glob("*.py")),
            self.config.coverage_policy,
        ]
        return {p.relative_to(self.config.root).as_posix(): digest(p) for p in paths}

    def initialize(self, admission_path, active_path):
        from transformers import AutoTokenizer

        self.tokenizers = tuple(
            AutoTokenizer.from_pretrained(path, local_files_only=True)
            for path in (admission_path, active_path)
        )

    def save_tokenizers(self, run):
        from huggingface_hub import snapshot_download

        paths = []
        for name, model, revision in (
            (
                "admission",
                self.config.policy["historical_admission_tokenizer"]["model_id"],
                self.config.policy["historical_admission_tokenizer"]["revision"],
            ),
            (
                "active",
                self.config.policy["model"]["model_id"],
                self.config.policy["model"]["revision"],
            ),
        ):
            target = run / "tokenizers" / name
            snapshot_download(
                model,
                revision=revision,
                local_dir=target,
                allow_patterns=[
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                    "vocab.json",
                    "merges.txt",
                    "special_tokens_map.json",
                    "config.json",
                ],
            )
            paths.append(str(target))
        return paths

    def construct(self, slot, alternative):
        seed = slot.seed(alternative)
        direction = tuple(Direction)[seed % 8]
        (anchor, _) = build_scene(
            SceneRequest(10, Direction.NORTH, "which", "sole"),
            identity=f"{self.config.anchor_prefix}-{seed}",
            family="anchor",
            structure_seed=seed,
        )
        family = story_key(anchor.objects, anchor.premise)
        (story, problem) = build_scene(
            SceneRequest(
                slot.level,
                direction,
                slot.kind,
                slot.scene_case,
                slot.count_value,
                variant=seed % 2,
            ),
            identity=slot.identity(alternative),
            family=f"{self.config.family_prefix}/{family}",
            structure_seed=seed,
            contingent=slot.contingent,
        )
        return (story, problem, family, direction)

    def preflight(self, slots, inventory, run):
        (jobs, reasons) = ([], Counter())
        with (run / "audit/constructions.jsonl").open("x") as journal:
            for slot in slots:
                selected = None
                for alternative in range(
                    self.config.policy["construction_candidates_per_slot"]
                ):
                    (_, problem, family, direction) = self.construct(slot, alternative)
                    keys = identities(problem)
                    reason = inventory.conflict(keys, slot.split, family)
                    reasons[reason or "selected"] += 1
                    append_json(
                        journal,
                        {
                            "slot_id": slot.identity(0),
                            "alternative": alternative,
                            "status": reason or "selected",
                            **keys,
                            "anchor_key": family,
                        },
                    )
                    if reason is None:
                        inventory.add(keys, slot.split, family)
                        selected = {
                            "slot": asdict(slot),
                            "alternative": alternative,
                            **keys,
                            "anchor_key": family,
                            "direction": direction.value,
                        }
                        jobs.append(selected)
                        break
                require(
                    selected is not None,
                    f"distinct supply exhausted for {slot.identity(0)}; no refill",
                )
                if len(jobs) % 100 == 0:
                    print(f"Preflight {len(jobs)}/{len(slots)}", flush=True)
        atomic_json(
            run / "audit/preflight.json",
            {"status": "passed", "selected": len(jobs), "attempts": dict(reasons)},
        )
        return jobs

    def prompt_budget(self, tokenizer, row, style):
        from spatial.v2.entailment_evaluation import stage2_suffix

        source = messages(row, style)
        prefix = tokenizer.apply_chat_template(
            source, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        suffix = tokenizer.encode(
            stage2_suffix(tokenizer, []), add_special_tokens=False
        )
        reserved = (
            len(ids)
            + self.config.policy["model"]["stage1_max_new_tokens"]
            + len(suffix)
            + self.config.policy["model"]["stage2_max_new_tokens"]
        )
        require(
            reserved <= self.config.policy["model"]["context_length"],
            "Stage-2 context reserve exceeded",
        )
        return {
            "id": row["id"],
            "style": style,
            "prompt_tokens": len(ids),
            "stage2_reserved_tokens": reserved,
        }

    def generate_one(self, job):
        slot = self.config.slot_type(**job["slot"])
        try:
            (admission, active) = self.tokenizers
            (story, problem, family, direction) = self.construct(
                slot, job["alternative"]
            )
            require(
                identities(problem) == {k: job[k] for k in ("query_key", "scene_key")}
                and family == job["anchor_key"],
                "preflight construction changed",
            )
            record = build_attempt(
                story,
                problem,
                slot.requested_depth,
                admission,
                omission_percent=100 if slot.case == "F" else 0,
                prompt_version=SHORT_VERSION,
                count_menu_values=self.config.policy.get("synthetic_count_menu_values")
                if slot.kind == "count"
                else None,
            )
            require(
                record["status"] == "accepted",
                f"admission {record['stage']}: {record.get('reason', record['status'])}",
            )
            rows = [
                r for r in record["rows"] if (r["answer"] == "F") == (slot.case == "F")
            ]
            require(len(rows) == 1, "expected one selected menu")
            record["rows"] = rows
            evidence = load_evidence(record)
            row = rows[0]
            expected_case = {
                "A-D": None,
                "F": "unique-unlisted",
                "ambiguous": "no-entailed-answer",
                "multiple": "multiple-entailed",
                "inconsistent": "inconsistent",
            }[slot.case]
            require(
                row["escape_reason"] == expected_case,
                "constructed semantic case differs",
            )
            require(
                slot.case != "A-D" or row["answer"] in "ABCD",
                "not a displayed sole answer",
            )
            if slot.kind == "count" and slot.case in ("A-D", "F"):
                require(
                    evidence.semantics.entailed == {slot.count_value},
                    "Count target changed",
                )
            if slot.contingent:
                require(
                    row["sole_entailed_with_contingent"],
                    "contingent alternative missing",
                )
            menu = list(row["menu"])
            if slot.answer_position is not None:
                old = "ABCD".index(row["answer"])
                new = "ABCD".index(slot.answer_position)
                (menu[old], menu[new]) = (menu[new], menu[old])
            solver = QuestionSolver()
            rebind_menu(row, evidence, menu, solver=solver)
            require(
                slot.answer_position is None or row["answer"] == slot.answer_position,
                "menu position allocation failed",
            )
            metadata = {
                "split": slot.split,
                "source": "controlled",
                "construction_level": slot.level,
                "verified_depth": record["measurement"]["depth"]
                if slot.cohort == "depth"
                else None,
                "cohort": slot.cohort,
                "case": slot.case,
                "direction": direction.value,
                "proof_pattern": "axis-split" if slot.level > 1 else "homogeneous",
                "nonhomogeneous_proof": slot.level > 1,
            }
            row.update(metadata)
            record.update(
                metadata,
                **{k: job[k] for k in ("query_key", "scene_key", "anchor_key")},
                slot=asdict(slot),
                alternative=job["alternative"],
            )
            if slot.cohort == "semantic":
                row["depth"] = None
                row["selected_explanation_depth_support"] = False
                row["substantive_depth_support"] = False
                row["decision_depth_support"] = False
            independent_plan_check(evidence, make_plan(evidence))
            audit_row(row, evidence, solver=solver)
            record = replay(record)
            row = record["rows"][0]
            if slot.case in ("A-D", "F", "multiple"):
                # Semantic-cohort positives still need the intended support;
                # this does not certify depth for contingent negative claims.
                require(
                    row["selected_explanation_depth"] == slot.level,
                    "selected positive support differs from construction level",
                )
            if slot.requested_depth is not None:
                require(
                    row["selected_explanation_depth_support"],
                    "selected explanation lacks required depth",
                )
            record["tokens"] = []
            (tokenized, chats) = ({}, {})
            for style in ("natural", "symbolic"):
                budget = self.prompt_budget(active, row, style)
                source = [
                    *messages(row, style),
                    {"role": "assistant", "content": row["references"][style]},
                ]
                (tokens, chat, stats) = tokenize(
                    active, source, row["answer"], self.config.policy["model"]
                )
                record["tokens"].append({**budget, **stats, "admitted": True})
                if slot.split in ("train", "dev"):
                    tokenized[style] = tokens
                    chats[style] = chat
            record["semantic_complexity_status"] = (
                "verified-semantics-no-depth-claim"
                if slot.cohort == "semantic"
                else "verified-answer-support"
            )
            return {
                "status": "passed",
                "record": record,
                "tokenized": tokenized,
                "chats": chats,
            }
        except (ValueError, RuntimeError, KeyError, TypeError, AssertionError) as exc:
            return {
                "status": "failed",
                "slot": asdict(slot),
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

    def external_one(self, source):
        """Keep source A-D menus; recompute Unique labels without trusting Corr gold."""
        import z3

        active = self.tokenizers[1]
        record = build_views(
            source, SelectionScope.ALL, omission_percent=0, prompt_version=SHORT_VERSION
        )
        require(len(record["rows"]) == 1, "external augmentation is forbidden")
        row = record["rows"][0]
        parsed_source = SpatialTextAdapter().parse(source["text"])
        (_, _, original) = adapt_problem(
            source["text"], SelectionScope.ALL, preserve_source_names=True
        )
        aliases = {name: f"Entity{i}" for (i, name) in enumerate(original.objects)}
        menu = []
        for letter in "ABCD":
            value = parsed_source.options[letter]
            if isinstance(original.query, DirectionQuery):
                direction = Direction.from_label(value)
                require(direction is not None, "invalid source direction option")
                menu.append(direction.value)
            elif isinstance(original.query, CountQuery):
                menu.append(int(value))
            else:
                name = value[4:] if value.lower().startswith("the ") else value
                require(name in aliases, "source Which option is not a scene object")
                menu.append(aliases[name])
        wrapper = {
            "interface": row["interface"],
            "problem": row["problem"],
            "semantics": record["semantics"],
            "evidence": record["evidence"],
            "rows": [row],
            "prompt_version": SHORT_VERSION,
            "scope_policy": row["scope_policy"],
            "study_scope": row["study_scope"],
        }
        evidence = load_evidence(wrapper) if record["evidence"] is not None else None
        solver = QuestionSolver()
        rebind_menu(row, evidence, menu, solver=solver, external=True)
        verify_view(row, solver=QuestionSolver())
        (problem, saved_menu, contract) = parse_row(row)
        (independent, claims) = independent_claims(problem)
        consistent = checked_sat(independent)
        entailed = {
            v
            for (v, claim) in claims.items()
            if consistent and (not checked_sat(independent, z3.Not(claim)))
        }
        require(
            row["answer"] == label(consistent, entailed, saved_menu, contract.value),
            "independent external gold differs",
        )
        if evidence is not None:
            audit_record(wrapper)
        row.update(
            dataset="SpatialMap-TQA-Corr",
            menu_origin="source-A-D",
            source_oracle_option=source.get("oracle_option"),
            unique_label_differs_from_source=row["answer"]
            != source.get("oracle_option"),
        )
        keys = identities(problem)
        return {
            "row": row,
            "evidence": {**record, "rows": [row]},
            "keys": keys,
            "tokens": [
                self.prompt_budget(active, row, style)
                for style in ("natural", "symbolic")
            ],
        }

    def source_inputs(self, args, run):
        dest = run / "inputs"
        projection = verify_source_export(
            args.original_source, args.corr_source, args.correction_receipt
        )
        require(
            len(projection) == self.config.policy["external_bases"],
            "wrong external source question count",
        )
        require(
            len({s["id"].rsplit(".", 1)[0] for s in projection})
            == self.config.policy["external_scenes"],
            "wrong external scene count",
        )
        for name, path in (
            ("original.jsonl", args.original_source),
            ("corr-export.jsonl", args.corr_source),
            ("correction-receipt.json", args.correction_receipt),
        ):
            shutil.copyfile(path, dest / name)
        with (dest / "corr-source.jsonl").open("x") as stream:
            for row in projection:
                append_json(stream, row)
        return projection

    def audit_inventory(self, records, slots, mode):
        require(len(records) == len(slots), "incomplete candidate")
        require(
            Counter(r["rows"][0]["base_id"] for r in records)
            == Counter(
                f"{s.identity(r['alternative'])}-{s.kind}"
                for (s, r) in zip(slots, records)
            ),
            "slot membership/order differs",
        )
        inventory = Inventory()
        for r, slot in zip(records, slots):
            require(r["slot"] == asdict(slot), "record has wrong slot")
            inventory.add(r, slot.split, r["anchor_key"])
        groups = defaultdict(list)
        for r in records:
            groups[r["split"], r["query_kind"]].append(r)
        if mode == "full":
            original = json.loads(self.config.coverage_policy.read_text())["policy"]
            for (split, kind), rows in groups.items():
                floor = (50 * self.config.policy["split_bases"][split] + 5999) // 6000
                for level in sorted({r["construction_level"] for r in rows}):
                    depth_rows = [
                        r
                        for r in rows
                        if r["cohort"] == "depth" and r["construction_level"] == level
                    ]
                    require(
                        len({r["family_id"] for r in depth_rows}) >= floor,
                        f"insufficient depth families: {split}/{kind}/{level}",
                    )
                require(
                    len({r["direction"] for r in rows}) == 8,
                    "missing direction coverage",
                )
                for rule in original["coverage_minima"]:
                    if rule["match"].get("query_kind", kind) != kind:
                        continue
                    if rule["match"].get("depth") not in (
                        None,
                        *{r["construction_level"] for r in rows},
                    ):
                        continue
                    translated = {**rule, "match": {**rule["match"], "split": split}}
                    if "query_kind" not in rule["match"]:
                        continue
                    found = [r for r in rows if matches(r, translated)]
                    for field in ("minimum_bases", "minimum_families"):
                        expected = (
                            rule.get(field, 0)
                            * self.config.policy["split_bases"][split]
                            + 5999
                        ) // 6000
                        actual = (
                            len(found)
                            if field == "minimum_bases"
                            else len({r["family_id"] for r in found})
                        )
                        require(
                            actual >= expected,
                            f"coverage failed {split}/{kind}: {translated['match']}",
                        )
        return {
            "status": "passed",
            "mode": mode,
            "records": len(records),
            "split_counts": dict(Counter(r["split"] for r in records)),
            "cases": dict(
                Counter(f"{r['split']}/{r['query_kind']}/{r['case']}" for r in records)
            ),
            "levels": dict(
                Counter(
                    f"{r['split']}/{r['query_kind']}/{r['construction_level']}"
                    for r in records
                )
            ),
        }

    def audit_exports(self, run, expected_counts):
        """Read published JSONL back, including gold-free inputs and mask sidecars."""
        for split, count in expected_counts.items():
            directory = run / "data" / split
            rows = read_jsonl(directory / "references.jsonl")
            require(
                len(rows) == len({r["id"] for r in rows}) == count,
                f"export count/identity mismatch: {split}",
            )
            require(all(r["split"] == split for r in rows), "export crossed splits")
            evidence = read_jsonl(directory / "evidence.jsonl")
            require(len(evidence) == count, "evidence count differs")
            for row, record in zip(rows, evidence, strict=True):
                require(
                    record["rows"] == [row], "evidence and reference exports differ"
                )
            for style in ("natural", "symbolic"):
                inputs = read_jsonl(directory / f"{style}-inputs.jsonl")
                require(
                    inputs
                    == [{"id": r["id"], "messages": messages(r, style)} for r in rows],
                    "model inputs changed or contain gold/audit fields",
                )
            if split not in ("train", "dev"):
                continue
            index = read_jsonl(directory / "index.jsonl")
            require(
                [r["id"] for r in index] == [r["id"] for r in rows],
                "token sidecar order differs",
            )
            for style in ("natural", "symbolic"):
                tokens = read_jsonl(directory / f"{style}-tokenized.jsonl")
                chats = read_jsonl(directory / f"{style}-chat.jsonl")
                for row, tok, chat, sidecar in zip(
                    rows, tokens, chats, index, strict=True
                ):
                    source = [
                        *messages(row, style),
                        {"role": "assistant", "content": row["references"][style]},
                    ]
                    (expected, native, _) = tokenize(
                        self.tokenizers[1],
                        source,
                        row["answer"],
                        self.config.policy["model"],
                    )
                    require(
                        tok == expected
                        and chat == {"id": row["id"], "messages": native},
                        "native token/chat export differs",
                    )
                    require(
                        hashlib.sha256(
                            json.dumps(tok, sort_keys=True).encode()
                        ).hexdigest()
                        == sidecar["token_sha256"][style],
                        "token sidecar hash differs",
                    )
        return {
            "status": "passed",
            "split_counts": expected_counts,
            "checks": [
                "export-readback",
                "gold-free-inputs",
                "native-mask-replay",
                "paired-order",
                "evidence-reference-equality",
            ],
        }

    def verify(self, run, *, require_training_ready=False):
        manifest = json.loads((run / "manifest.json").read_text())
        receipt = json.loads((run / "generation.json").read_text())
        require(
            manifest["schema"] == self.config.manifest_schema, "wrong manifest schema"
        )
        require(
            manifest["status"] in ("candidate-audited", "smoke-passed"),
            "generation did not finish successfully",
        )
        require(
            receipt["status"] == manifest["status"]
            and receipt["manifest_sha256"] == digest(run / "manifest.json"),
            "generation completion receipt differs",
        )
        for name, sha in manifest["payload_sha256"].items():
            path = (run / name).resolve()
            require(
                path.is_relative_to(run.resolve()) and path.is_file(),
                f"missing or invalid payload: {name}",
            )
            require(digest(path) == sha, f"payload changed: {name}")
        require(
            manifest["payload_sha256"] == payload_hashes(run),
            "payload inventory changed",
        )
        require(
            manifest["source_sha256"] == self.fingerprints(),
            "generation code/input policy changed",
        )
        if require_training_ready:
            require(
                manifest["training_ready"],
                "candidate is not a training release: "
                + ", ".join(manifest["pending_release_checks"]),
            )
        return manifest

    def generate(self, args):
        run = args.run_dir.resolve()
        require(not run.exists(), "run directory exists; use a fresh path")
        require(run.parent.is_dir(), "run parent must exist")
        run.mkdir()
        for name in ("inputs", "data", "audit", "tokenizers"):
            (run / name).mkdir()
        receipt = run / "generation.json"
        atomic_json(
            receipt, {"status": "running", "mode": args.mode, "training_ready": False}
        )
        hashes = self.fingerprints()
        try:
            policy = json.loads(json.dumps(self.config.policy))
            slots = [self.config.slot_type(**s) for s in policy["slots"]]
            if args.mode == "smoke":
                slots = list(self.config.smoke_slots)
            policy["mode"] = args.mode
            policy["selected_slots"] = [asdict(s) for s in slots]
            atomic_json(run / "inputs/generation-policy.json", policy)
            sources = self.source_inputs(args, run)
            paths = self.save_tokenizers(run)
            self.initialize(*paths)
            inventory = Inventory()
            external = []
            for source in sources:
                (problem, _, _) = adapt_problem(
                    source["text"], SelectionScope.ALL, preserve_source_names=True
                )
                from spatial.v2.entailment_scene_protocol import all_scene_problem

                problem = all_scene_problem(problem)
                inventory.add(
                    identities(problem), "external-test", allow_external_duplicate=True
                )
            selected_sources = sources
            if args.mode == "smoke":
                selected_sources = [
                    next(s for s in sources if s["id"].endswith(f".{i}"))
                    for i in range(3)
                ]
            with ExitStack() as stack:
                handles = open_split(stack, run, "external-test")
                for source in selected_sources:
                    item = self.external_one(source)
                    external.append(item)
                    row = item["row"]
                    append_json(handles["references"], row)
                    append_json(handles["evidence"], item["evidence"])
                    for measurement in item["tokens"]:
                        append_json(handles["token-measurements"], measurement)
                    for style in ("natural", "symbolic"):
                        append_json(
                            handles[f"{style}-inputs"],
                            {"id": row["id"], "messages": messages(row, style)},
                        )
                    if len(external) % 100 == 0:
                        print(
                            f"External audited {len(external)}/{len(selected_sources)}",
                            flush=True,
                        )
            atomic_json(
                run / "audit/external.json",
                {
                    "status": "passed",
                    "bases": len(external),
                    "source_menus_preserved": True,
                    "unique_label_changes": sum(
                        i["row"]["unique_label_differs_from_source"] for i in external
                    ),
                    "query_counts": dict(
                        Counter(i["row"]["query_kind"] for i in external)
                    ),
                },
            )
            jobs = self.preflight(slots, inventory, run)
            records = []
            with ExitStack() as stack:
                handles = {
                    split: open_split(stack, run, split)
                    for split in self.config.policy["split_bases"]
                }
                journal = stack.enter_context(
                    (run / "audit/admissions.jsonl").open("x")
                )
                pool = stack.enter_context(
                    ProcessPoolExecutor(
                        max_workers=args.workers,
                        mp_context=multiprocessing.get_context("spawn"),
                        initializer=initialize_worker,
                        initargs=(self.config, *paths),
                    )
                )
                for offset in range(0, len(jobs), args.workers):
                    batch = jobs[offset : offset + args.workers]
                    for job, result in zip(
                        batch, pool.map(generate_worker, batch), strict=True
                    ):
                        append_json(
                            journal,
                            {
                                "slot_id": self.config.slot_type(
                                    **job["slot"]
                                ).identity(job["alternative"]),
                                **{
                                    k: v
                                    for (k, v) in result.items()
                                    if k not in ("record", "tokenized", "chats")
                                },
                            },
                        )
                        require(
                            result["status"] == "passed",
                            f"generation stopped, no refill: {result.get('reason')}",
                        )
                        record = result["record"]
                        write_synthetic(handles[record["split"]], result)
                        records.append(record)
                        if len(records) % 50 == 0:
                            print(
                                f"Audited {len(records)}/{len(slots)} synthetic questions",
                                flush=True,
                            )
            summary = self.audit_inventory(records, slots, args.mode)
            atomic_json(run / "audit/summary.json", summary)
            counts = {**summary["split_counts"], "external-test": len(external)}
            atomic_json(run / "audit/exports.json", self.audit_exports(run, counts))
            samples, selection = select_review_records(records)
            with (run / "audit/review-samples.jsonl").open("x") as stream:
                for record in samples:
                    append_json(stream, record["rows"][0])
            atomic_json(run / "audit/review-selection.json", selection)
            require(hashes == self.fingerprints(), "source changed during generation")
            manifest = {
                "schema": self.config.manifest_schema,
                "source_inventory_version": "script-metadata-generation-v3",
                "status": "candidate-audited"
                if args.mode == "full"
                else "smoke-passed",
                "mode": args.mode,
                "model": self.config.policy["model"],
                "training_ready": False,
                "pending_release_checks": list(self.config.pending_release_checks),
                "source_sha256": hashes,
                "payload_sha256": payload_hashes(run),
                "environment": {
                    "python": platform.python_version(),
                    **{name: version(name) for name in DEPENDENCIES},
                },
                "split_counts": counts,
                "tokenization": "native-thinking-pretokenized-v1",
                "training_order": "file-order; same IDs in both styles",
            }
            atomic_json(run / "manifest.json", manifest)
            atomic_json(
                receipt,
                {
                    "status": manifest["status"],
                    "training_ready": False,
                    "manifest_sha256": digest(run / "manifest.json"),
                },
            )
            self.verify(run)
            print(
                json.dumps(
                    {
                        "run": str(run),
                        "status": manifest["status"],
                        "training_ready": False,
                        "counts": manifest["split_counts"],
                    }
                ),
                flush=True,
            )
        except BaseException as exc:
            atomic_json(
                receipt,
                {
                    "status": "failed",
                    "training_ready": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            if (run / "manifest.json").exists():
                (run / "manifest.json").rename(run / "audit/failed-manifest.json")
            raise


def main(config, argv=None):
    """Run the shared build/verify CLI with an experiment-supplied configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--run-dir", type=Path, required=True)
    build.add_argument("--mode", choices=("full", "smoke"), default="full")
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--original-source", type=Path, default=config.original_source)
    build.add_argument("--corr-source", type=Path, default=config.corr_source)
    build.add_argument(
        "--correction-receipt", type=Path, default=config.correction_receipt
    )
    check = commands.add_parser("verify")
    check.add_argument("--run-dir", type=Path, required=True)
    check.add_argument("--require-training-ready", action="store_true")
    args = parser.parse_args(argv)
    workflow = GenerationWorkflow(config)
    if args.command == "verify":
        result = workflow.verify(
            args.run_dir.resolve(), require_training_ready=args.require_training_ready
        )
        print(
            json.dumps(
                {"status": result["status"], "training_ready": result["training_ready"]}
            )
        )
    else:
        require(1 <= args.workers <= (os.cpu_count() or 1), "invalid worker count")
        workflow.generate(args)
