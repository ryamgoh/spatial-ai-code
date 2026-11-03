"""Run-local candidate generation with explicit experiment policy and paths."""

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import random
import shutil
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from spatial.v2.composition_balance import query_profile
from spatial.v2.composition_feasibility import stable_int
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
from spatial.v2.entailment_generator import (
    EXPLICIT_SIZE_VERSION,
    SceneRequest,
    build_scene,
)
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


def initialize_worker(config, tokenizer_path=None):
    global _WORKER
    _WORKER = GenerationWorkflow(config)
    if tokenizer_path is not None:
        _WORKER.initialize(tokenizer_path)


def construct_worker(slot, alternative):
    require(_WORKER is not None, "construction worker was not initialized")
    return _WORKER.construct_candidate(slot, alternative)


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
        self.query_splits = {}
        self.scenes = {}
        self.families = {}

    def conflict(self, keys, split, family=None):
        if (
            keys["query_key"] in self.query_splits
            and self.query_splits[keys["query_key"]] != split
        ):
            return "duplicate-query"
        if keys["scene_key"] in self.scenes and self.scenes[keys["scene_key"]] != split:
            return "cross-split-scene"
        if family in self.families and self.families[family] != split:
            return "cross-split-family"
        if keys["query_key"] in self.query_splits:
            return "duplicate-query"
        return None

    def training_duplicate_allowed(self, keys, split, family=None):
        return (
            split == "train"
            and self.query_splits.get(keys["query_key"]) == "train"
            and self.scenes.get(keys["scene_key"], "train") == "train"
            and self.families.get(family, "train") == "train"
        )

    def add(
        self,
        keys,
        split,
        family=None,
        *,
        allow_external_duplicate=False,
        allow_training_duplicate=False,
    ):
        reason = self.conflict(keys, split, family)
        require(
            reason is None
            or (
                allow_external_duplicate
                and split == "external-test"
                and reason == "duplicate-query"
                and self.query_splits.get(keys["query_key"]) == split
                and self.scenes.get(keys["scene_key"], split) == split
                and self.families.get(family, split) == split
            )
            or (
                allow_training_duplicate
                and reason == "duplicate-query"
                and self.training_duplicate_allowed(keys, split, family)
            ),
            f"inventory conflict: {reason}",
        )
        self.query_splits[keys["query_key"]] = split
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
        self.tokenizer = None

    def fingerprints(self):
        paths = [
            *sorted((self.config.root / "spatial/v2").glob("*.py")),
            *sorted(self.config.experiment.glob("*.py")),
            self.config.coverage_policy,
        ]
        return {p.relative_to(self.config.root).as_posix(): digest(p) for p in paths}

    def initialize(self, tokenizer_path):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True
        )

    def save_tokenizer(self, run):
        from huggingface_hub import snapshot_download

        require(
            self.config.policy.get("admission_tokenizer_policy")
            == "pinned-training-tokenizer-v1",
            "unsupported admission tokenizer policy",
        )
        model = self.config.policy["model"]
        target = run / "tokenizers" / "active"
        snapshot_download(
            model["model_id"],
            revision=model["revision"],
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
        return str(target)

    def sample_size(self, slot, alternative):
        size_policy = self.config.policy.get("size_policy")
        if size_policy is None:
            return None
        require(
            self.config.policy["scene_policy_version"] == EXPLICIT_SIZE_VERSION
            and size_policy["version"] == "linear-coupled-sizes-v1",
            "unknown explicit size policy",
        )
        bounds = size_policy["levels"][str(slot.level)]
        low, high = size_policy["premises_minus_objects"]
        pairs = [
            (n, n + extra)
            for n in range(bounds["objects"][0], bounds["objects"][1] + 1)
            for extra in range(low, high + 1)
        ]
        require(
            pairs
            and all(
                bounds["premises"][0] <= p <= bounds["premises"][1] for _, p in pairs
            ),
            "size table disagrees with coupled sampling",
        )
        seed = stable_int(f"{size_policy['seed_namespace']}/{slot.seed(alternative)}")
        return random.Random(seed).choice(pairs)

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
            varied_context=self.config.policy.get("varied_context", False),
            size_pair=self.sample_size(slot, alternative),
            augmentation_limit=self.config.policy.get(
                "augmentation_candidate_limit", 512
            ),
        )
        return (story, problem, family, direction)

    def construct_candidate(self, slot, alternative):
        _, problem, family, direction = self.construct(slot, alternative)
        return {
            **identities(problem),
            "anchor_key": family,
            "direction": direction.value,
        }

    def preflight(self, slots, inventory, run, *, workers=1):
        require(type(workers) is int and workers > 0, "workers must be positive")
        (jobs, reasons) = ([], Counter())
        with ExitStack() as stack:
            journal = stack.enter_context((run / "audit/constructions.jsonl").open("x"))
            pool = None
            pending = {}
            if workers > 1 and slots:
                pool = ProcessPoolExecutor(
                    max_workers=min(workers, len(slots)),
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_worker,
                    initargs=(self.config,),
                )
                stack.callback(pool.shutdown, wait=True, cancel_futures=True)
                pending = {
                    index: pool.submit(construct_worker, slot, 0)
                    for index, slot in enumerate(slots[:workers])
                }
            for index, slot in enumerate(slots):
                selected = None
                for alternative in range(
                    self.config.policy["construction_candidates_per_slot"]
                ):
                    if pool is None:
                        candidate = self.construct_candidate(slot, alternative)
                    elif alternative == 0:
                        candidate = pending.pop(index).result()
                    else:
                        candidate = pool.submit(
                            construct_worker, slot, alternative
                        ).result()
                    keys = {k: candidate[k] for k in ("query_key", "scene_key")}
                    family = candidate["anchor_key"]
                    reason = inventory.conflict(keys, slot.split, family)
                    duplicate = (
                        reason == "duplicate-query"
                        and alternative
                        == self.config.policy["construction_candidates_per_slot"] - 1
                        and self.config.policy.get("training_duplicate_policy")
                        == "last-attempt-within-train-only-v1"
                        and inventory.training_duplicate_allowed(
                            keys, slot.split, family
                        )
                    )
                    status = (
                        "selected-training-duplicate"
                        if duplicate
                        else reason or "selected"
                    )
                    reasons[status] += 1
                    append_json(
                        journal,
                        {
                            "slot_id": slot.identity(0),
                            "alternative": alternative,
                            "status": status,
                            **keys,
                            "anchor_key": family,
                        },
                    )
                    if reason is None or duplicate:
                        inventory.add(
                            keys, slot.split, family, allow_training_duplicate=duplicate
                        )
                        selected = {
                            "slot": asdict(slot),
                            "alternative": alternative,
                            **keys,
                            "anchor_key": family,
                            "direction": candidate["direction"],
                            "training_duplicate": duplicate,
                        }
                        jobs.append(selected)
                        break
                require(
                    selected is not None,
                    f"distinct supply exhausted for {slot.identity(0)}; no refill",
                )
                # Refill only after ordered acceptance; retries never mutate
                # ownership ahead of an earlier slot, regardless of finish order.
                following = index + workers
                if pool is not None and following < len(slots):
                    pending[following] = pool.submit(
                        construct_worker, slots[following], 0
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

    def admission_tokens(self, tokenizer, row, style):
        """Apply the same native-thinking checks used by final token exports."""
        result = {"id": row["id"], "style": style}
        try:
            result.update(self.prompt_budget(tokenizer, row, style))
            source = [
                *messages(row, style),
                {"role": "assistant", "content": row["references"][style]},
            ]
            _, _, stats = tokenize(
                tokenizer, source, row["answer"], self.config.policy["model"]
            )
            result.update(stats, admitted=True)
        except ValueError as exc:
            result.update(admitted=False, reason=str(exc))
        return result

    def generate_one(self, job):
        slot = self.config.slot_type(**job["slot"])
        token_rejections = []
        try:
            active = self.tokenizer
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
                active,
                omission_percent=100 if slot.case == "F" else 0,
                prompt_version=SHORT_VERSION,
                count_menu_values=self.config.policy.get("synthetic_count_menu_values")
                if slot.kind == "count"
                else None,
                measure_tokens=self.admission_tokens,
            )
            token_rejections = [
                t for t in record.get("tokens", ()) if not t["admitted"]
            ]
            rejection_reason = record.get("reason", record["status"])
            if token_rejections:
                rejection_reason += ": " + "; ".join(
                    f"{t['style']}: {t['reason']}" for t in token_rejections
                )
            require(
                record["status"] == "accepted",
                f"admission {record['stage']}: {rejection_reason}",
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
            if (
                slot.case == "multiple"
                and "multiple_which_entailed_objects" in self.config.policy
            ):
                require(
                    len(evidence.semantics.entailed)
                    == self.config.policy["multiple_which_entailed_objects"],
                    "Which multiplicity changed",
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
                training_duplicate=job.get("training_duplicate", False),
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
            size_pair = self.sample_size(slot, job["alternative"])
            if size_pair is not None:
                require(
                    (len(problem.objects), len(problem.premise.operands)) == size_pair,
                    "constructed scene differs from requested size",
                )
                record["requested_size"] = dict(zip(("objects", "premises"), size_pair))
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
                **({"token_rejections": token_rejections} if token_rejections else {}),
            }

    def external_one(self, source):
        """Keep source A-D menus; recompute Unique labels without trusting Corr gold."""
        import z3

        active = self.tokenizer
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
            duplicate = r.get("training_duplicate", False)
            if duplicate:
                require(
                    self.config.policy.get("training_duplicate_policy")
                    == "last-attempt-within-train-only-v1"
                    and r["alternative"]
                    == self.config.policy["construction_candidates_per_slot"] - 1
                    and inventory.training_duplicate_allowed(
                        r, slot.split, r["anchor_key"]
                    ),
                    "invalid training duplicate fallback",
                )
            size_pair = self.sample_size(slot, r["alternative"])
            if size_pair is not None:
                requested = dict(zip(("objects", "premises"), size_pair))
                require(
                    r["requested_size"] == requested
                    and all(
                        r["graph"][key] == value for key, value in requested.items()
                    ),
                    "inventory scene size differs from policy",
                )
            inventory.add(
                r, slot.split, r["anchor_key"], allow_training_duplicate=duplicate
            )
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
            "scene_size_counts": dict(
                Counter(
                    f"{r['split']}/{r['query_kind']}/{r['construction_level']}/"
                    f"{r['case']}/{r['graph']['objects']}/{r['graph']['premises']}"
                    for r in records
                )
            ),
            "training_duplicates_accepted": sum(
                r.get("training_duplicate", False) for r in records
            ),
            "canonical_query_frequencies": {
                split: dict(
                    Counter(r["query_key"] for r in records if r["split"] == split)
                )
                for split in self.config.policy["split_bases"]
            },
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
        export_splits = self.config.policy.get("export_splits", {})
        groups = defaultdict(dict)
        for split, count in expected_counts.items():
            groups[export_splits.get(split, split)][split] = count
        for split, subsets in groups.items():
            count = sum(subsets.values())
            directory = run / "data" / split
            rows = read_jsonl(directory / "references.jsonl")
            require(
                len(rows) == len({r["id"] for r in rows}) == count,
                f"export count/identity mismatch: {split}",
            )
            require(
                Counter(r["split"] for r in rows) == subsets,
                "export subgroup counts differ",
            )
            evidence = read_jsonl(directory / "evidence.jsonl")
            require(len(evidence) == count, "evidence count differs")
            for row, record in zip(rows, evidence, strict=True):
                require(
                    record["rows"] == [row], "evidence and reference exports differ"
                )
                if split != "external-test":
                    require(
                        record["split"] == row["split"], "evidence subgroup differs"
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
                        self.tokenizer,
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
            "split_counts": {
                split: sum(subsets.values()) for split, subsets in groups.items()
            },
            "generation_split_counts": expected_counts,
            "checks": [
                "export-readback",
                "subgroup-counts",
                "gold-free-inputs",
                "native-mask-replay",
                "paired-order",
                "evidence-reference-equality",
            ],
        }

    def verify(self, run, *, require_training_ready=False):
        return self.verify_saved(
            run,
            self.fingerprints(),
            self.config.manifest_schema,
            require_training_ready=require_training_ready,
        )

    @staticmethod
    def verify_saved(run, fingerprints, schema, *, require_training_ready=False):
        """Verify a frozen package against an explicitly supplied source inventory."""
        manifest = json.loads((run / "manifest.json").read_text())
        receipt = json.loads((run / "generation.json").read_text())
        require(manifest["schema"] == schema, "wrong manifest schema")
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
            manifest["source_sha256"] == fingerprints,
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
            tokenizer_path = self.save_tokenizer(run)
            self.initialize(tokenizer_path)
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
            jobs = self.preflight(slots, inventory, run, workers=args.workers)
            records = []
            export_splits = self.config.policy.get("export_splits", {})
            with ExitStack() as stack:
                handles = {
                    split: open_split(stack, run, split)
                    for split in dict.fromkeys(
                        export_splits.get(s, s)
                        for s in self.config.policy["split_bases"]
                    )
                }
                journal = stack.enter_context(
                    (run / "audit/admissions.jsonl").open("x")
                )
                pool = stack.enter_context(
                    ProcessPoolExecutor(
                        max_workers=args.workers,
                        mp_context=multiprocessing.get_context("spawn"),
                        initializer=initialize_worker,
                        initargs=(self.config, tokenizer_path),
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
                        destination = export_splits.get(
                            record["split"], record["split"]
                        )
                        write_synthetic(handles[destination], result)
                        records.append(record)
                        if len(records) % 50 == 0:
                            print(
                                f"Audited {len(records)}/{len(slots)} synthetic questions",
                                flush=True,
                            )
            summary = self.audit_inventory(records, slots, args.mode)
            atomic_json(run / "audit/summary.json", summary)
            counts = {**summary["split_counts"], "external-test": len(external)}
            export_audit = self.audit_exports(run, counts)
            atomic_json(run / "audit/exports.json", export_audit)
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
                "split_counts": export_audit["split_counts"],
                "generation_split_counts": counts,
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
    release = commands.add_parser("release")
    release.add_argument("--run-dir", type=Path, required=True)
    release.add_argument("--evidence", type=Path, required=True)
    release.add_argument("--generation-revision", required=True)
    release_check = commands.add_parser("verify-release")
    release_check.add_argument("--run-dir", type=Path, required=True)
    release_check.add_argument("--release", type=Path, required=True)
    release_audit = commands.add_parser("audit-release")
    release_audit.add_argument("--run-dir", type=Path, required=True)
    release_audit.add_argument("--evidence", type=Path, required=True)
    release_audit.add_argument("--generation-revision", required=True)
    release_audit.add_argument("--workers", type=int, default=4)
    evaluation_check = commands.add_parser("verify-evaluation")
    evaluation_check.add_argument("--run", type=Path, required=True)
    evaluation_check.add_argument("--training-revision")
    args = parser.parse_args(argv)
    if args.command == "verify-evaluation":
        from spatial.v2.entailment_final_evaluation import verify_campaign

        verify_campaign(args.run.resolve(), training_revision=args.training_revision)
        print("Selections, evaluation policy and all prompt budgets verified.")
    elif args.command == "audit-release":
        from spatial.v2.entailment_release import verify_candidate
        from spatial.v2.entailment_release_checks import (
            data_profile,
            diagnose,
            replay_candidate,
        )

        artifacts, evidence = args.run_dir.resolve(), args.evidence.resolve()
        require(
            not evidence.is_relative_to(artifacts), "evidence must be outside candidate"
        )
        require(1 <= args.workers <= (os.cpu_count() or 1), "invalid worker count")
        verify_candidate(
            artifacts, config.root, config.manifest_schema, args.generation_revision
        )
        require(
            not any(
                (evidence / name).exists()
                for name in ("diagnostics.json", "replay.json", "profile.json")
            ),
            "audit evidence exists; use a fresh directory",
        )
        evidence.mkdir(parents=True, exist_ok=True)
        atomic_json(evidence / "diagnostics.json", diagnose(artifacts))
        atomic_json(evidence / "profile.json", data_profile(artifacts))
        atomic_json(evidence / "replay.json", replay_candidate(artifacts, args.workers))
        verify_candidate(
            artifacts, config.root, config.manifest_schema, args.generation_revision
        )
        print(
            json.dumps(
                {
                    "status": "automated-checks-complete",
                    "paired_review": "still-required",
                }
            )
        )
    elif args.command in ("release", "verify-release"):
        from spatial.v2.entailment_release import seal_release, verify_release

        if args.command == "release":
            receipt = seal_release(
                config, args.run_dir, args.evidence, args.generation_revision
            )
        else:
            receipt = args.release
            verify_release(
                args.run_dir, receipt, root=config.root, schema=config.manifest_schema
            )
        print(
            json.dumps(
                {
                    "status": "data-ready",
                    "release": str(receipt),
                    "runtime_checks": "required-inside-SFT",
                }
            )
        )
    elif args.command == "verify":
        result = GenerationWorkflow(config).verify(
            args.run_dir.resolve(), require_training_ready=args.require_training_ready
        )
        print(
            json.dumps(
                {"status": result["status"], "training_ready": result["training_ready"]}
            )
        )
    else:
        require(1 <= args.workers <= (os.cpu_count() or 1), "invalid worker count")
        GenerationWorkflow(config).generate(args)
