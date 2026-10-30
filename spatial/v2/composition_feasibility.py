"""Fixed depth/semantic feasibility schedule; never exports training or a seal."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

from spatial.v2.composition_audit import aligned_pair
from spatial.v2.composition_contracts import (
    UnsupportedEvidence,
    VerificationError,
    analyze,
    encode,
    parse_prompt,
    render_prompt,
)
from spatial.v2.composition_dataset import random_stories, rename_story
from spatial.v2.composition_depth import measure_depth, signature
from spatial.v2.composition_evidence import build_evidence, render_pair
from spatial.v2.composition_pilot import (
    Story,
    construct_stories,
    menus_for,
    renamed_problem,
    story_key,
)
from spatial.v2.composition_training import (
    CONTEXT,
    REVISION,
    STAGE1_CAP,
    completion_suffix,
    digest,
    prompt_text,
    tokenize_example,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import (
    And,
    Direction,
    SpatialSolverV2,
    WhichQuery,
    direction_constraint,
    direction_signs,
)

SEED = 20261101
ROLES = ("train-pilot", "dev-pilot", "in-range-test-pilot")


@dataclass(frozen=True)
class Attempt:
    story: Story
    role: str
    requested_depth: int | None


def stable_int(value):
    return int(hashlib.sha256(value.encode()).hexdigest(), 16)


def scramble(story, rng):
    story = rename_story(story, rng)
    order = list(range(len(story.premise.operands)))
    rng.shuffle(order)
    atoms = []
    for i in order:
        atom = story.premise.operands[i]
        if rng.randrange(2):
            signs = direction_signs(next(iter(atom.allowed)))
            inverse = next(
                d for d in Direction if direction_signs(d) == tuple(-s for s in signs)
            )
            atom = direction_constraint(atom.reference, atom.subject, inverse)
        atoms.append(atom)
    return replace(
        story,
        premise=And(tuple(atoms)),
        distractors=tuple(j for j, i in enumerate(order) if i in story.distractors),
    )


def depth_story(depth, sample, rng):
    role_index, variant = divmod(sample, 2)
    count = 1 if variant == 0 else (0, 2, 3, 4, 5)[(depth + role_index) % 5]
    queried = rng.choice(tuple(Direction))
    desired = (
        queried
        if count
        else next(
            d
            for d in Direction
            if direction_signs(d) == tuple(-s for s in direction_signs(queried))
        )
    )
    signs = direction_signs(desired)
    compatible = [
        d
        for d in Direction
        if all(got in (0, want) for got, want in zip(direction_signs(d), signs))
    ]
    candidates = tuple(f"C{i}" for i in range(5))
    path = ["O", *(f"L{i}" for i in range(depth - 1)), "C0"]
    objects = list(dict.fromkeys([*path, *candidates]))
    atoms = []
    for i, (reference, subject) in enumerate(pairwise(path)):
        # One fully strict/equal requested edge fixes the endpoint's exact sign.
        direction = desired if i == 0 else rng.choice(compatible)
        atoms.append(direction_constraint(subject, reference, direction))
    excluded = next(
        d
        for d in Direction
        if direction_signs(d) == tuple(-s for s in direction_signs(queried))
    )
    for i in range(1, 5):
        atoms.append(
            direction_constraint(candidates[i], "O", queried if i < count else excluded)
        )
    distractors = []
    for i in range(rng.randrange(4)):
        obj = f"Branch{i}"
        objects.append(obj)
        distractors.append(len(atoms))
        atoms.append(
            direction_constraint(obj, rng.choice(path), rng.choice(tuple(Direction)))
        )
    for i in range(rng.randrange(3)):
        a, b = f"Extra{i}A", f"Extra{i}B"
        objects.extend((a, b))
        distractors.append(len(atoms))
        atoms.append(direction_constraint(a, b, rng.choice(tuple(Direction))))
    band = "in-range" if depth <= 8 else "extrapolation"
    family = f"depth-{band}-family-{sample}"
    story = Story(
        f"depth-{depth}-sample-{sample}",
        family,
        "controlled-depth",
        tuple(objects),
        And(tuple(atoms)),
        "C0",
        "O",
        candidates,
        queried,
        tuple(distractors),
    )
    return scramble(story, rng)


def schedule(seed=SEED):
    rng = random.Random(seed)
    attempts = [
        Attempt(
            depth_story(depth, sample, rng),
            ROLES[sample // 2] if depth <= 8 else "extrapolation-pilot",
            depth,
        )
        for depth in range(1, 11)
        for sample in range(6)
    ]
    controls = (
        *random_stories(seed + 1, world_count=12, atom_count=12),
        *construct_stories(seed + 2, 1),
    )
    for story in controls:
        role = ROLES[stable_int(story.source_family) % len(ROLES)]
        attempts.append(Attempt(scramble(story, rng), role, None))
    return tuple(attempts)


def split_conflicts(attempts):
    """Quarantine entire source families on cross-role identity collisions."""
    key_roles, family_roles = defaultdict(set), defaultdict(set)
    keys = {}
    for attempt in attempts:
        story = attempt.story
        key = story_key(story.objects, story.premise)
        keys[story.story_id] = key
        key_roles[key].add(attempt.role)
        family_roles[story.source_family].add(attempt.role)
    bad = {
        a.story.source_family
        for a in attempts
        if len(key_roles[keys[a.story.story_id]]) > 1
        or len(family_roles[a.story.source_family]) > 1
    }
    return keys, bad


def verify_transformations(story, problem, semantics, solver):
    variant, names = renamed_problem(problem)
    changed = analyze(variant, solver=solver, include_witnesses=False)
    expected = signature(semantics)
    if isinstance(problem.query, WhichQuery):
        expected = (
            semantics.consistent,
            frozenset(names[x] for x in semantics.possible),
            frozenset(names[x] for x in semantics.certain),
        )
    if signature(changed) != expected:
        raise VerificationError("rename/order changed full semantics")
    for removed in [*((i,) for i in story.distractors), story.distractors]:
        if not removed:
            continue
        reduced = replace(
            problem,
            premise=And(
                tuple(
                    atom
                    for i, atom in enumerate(problem.premise.operands)
                    if i not in removed
                )
            ),
        )
        if signature(
            analyze(reduced, solver=solver, include_witnesses=False)
        ) != signature(semantics):
            raise VerificationError("distractor removal changed full semantics")


def token_measurement(tokenizer, row, style):
    """Keep lengths on failures without weakening the shared admission guard."""
    prefix = prompt_text(tokenizer, row, style)
    target = row["references"][style]
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(
        prefix + target.strip() + "<|im_end|>\n", add_special_tokens=False
    )
    result = {
        "id": row["id"],
        "style": style,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "training_tokens": len(full_ids),
        "supervised_tokens": len(full_ids) - len(prompt_ids),
        "stage2_reserved_tokens": len(prompt_ids)
        + STAGE1_CAP
        + 16
        + len(tokenizer.encode(completion_suffix(), add_special_tokens=False)),
    }
    try:
        tokenize_example(tokenizer, row, style)
        if result["stage2_reserved_tokens"] > CONTEXT:
            raise ValueError("Stage-2 reservation exceeds context")
    except ValueError as exc:
        result.update(admitted=False, reason=str(exc))
    else:
        result["admitted"] = True
    return result


def graph_profile(story):
    adjacent = {obj: set() for obj in story.objects}
    for atom in story.premise.operands:
        adjacent[atom.subject].add(atom.reference)
        adjacent[atom.reference].add(atom.subject)
    return {
        "objects": len(story.objects),
        "premises": len(story.premise.operands),
        "candidates": len(story.candidates),
        "verified_distractors": len(story.distractors),
        "degree_sequence": sorted(len(v) for v in adjacent.values()),
    }


def build_base(attempt, problem, tokenizer, solver):
    story = attempt.story
    kind = type(problem.query).__name__.removesuffix("Query").lower()
    base_id = f"{story.story_id}-{kind}"
    evidence = build_evidence(problem, solver=solver)
    semantics = evidence.semantics
    verify_transformations(story, problem, semantics, solver)
    try:
        measured = measure_depth(problem, solver=solver)
    except UnsupportedEvidence as exc:
        measured = {"depth": None, "reason": str(exc)}
    if (
        attempt.requested_depth is not None
        and measured["depth"] != attempt.requested_depth
    ):
        raise UnsupportedEvidence(
            f"requested depth {attempt.requested_depth}, measured {measured}"
        )
    menus = menus_for(semantics, stable_int(base_id))
    if stable_int(base_id) % 4:
        menus = menus[:1]
    rows, tokens = [], []
    for i, menu in enumerate(menus):
        for mode in AnswerMode:
            prompt = render_prompt(problem, menu, mode)
            if parse_prompt(prompt) != (problem, menu, mode):
                raise VerificationError("prompt round trip changed semantics")
            decision = encode(semantics, menu, mode)
            pair = render_pair(
                evidence, menu, mode, solver=solver, compact_unique_direction=True
            )
            row = {
                "id": f"{base_id}-m{i}-{mode.value}",
                "base_id": base_id,
                "story_id": story.story_id,
                "family_id": story.source_family,
                "split": attempt.role,
                "recipe": story.recipe,
                "query_kind": kind,
                "depth": measured["depth"],
                "contract": mode.value,
                "semantic_case": semantics.case,
                "possible_count": len(semantics.possible),
                "displayed_possible_count": len(semantics.possible & set(menu)),
                "answer": decision.letters,
                "escape_reason": decision.escape_reason,
                "returned_count": None
                if decision.letters in {"E", "F"}
                else len(decision.letters),
                "coverage": "full" if semantics.possible <= set(menu) else "partial",
                "menu": menu,
                "prompt": prompt,
                "references": pair,
            }
            aligned_pair(row)
            rows.append(row)
            tokens.extend(token_measurement(tokenizer, row, style) for style in pair)
    return {
        "base_id": base_id,
        "requested_depth": attempt.requested_depth,
        "measurement": measured,
        "graph": graph_profile(story),
        "evidence": tagged_dataclass_to_dict(evidence),
        "rows": rows,
        "tokens": tokens,
    }


def summarize(records):
    accepted = [r for r in records if r["status"] == "accepted"]
    rows = [row for r in accepted for row in r["data"]["rows"]]
    all_tokens = [t for r in records if "data" in r for t in r["data"]["tokens"]]
    groups = {}
    for fields in (
        ("split", "query_kind", "depth"),
        ("query_kind", "contract", "semantic_case", "returned_count"),
        ("split", "contract", "answer"),
        ("contract", "escape_reason"),
    ):
        groups["/".join(fields)] = dict(
            sorted(
                Counter("/".join(str(row[f]) for f in fields) for row in rows).items()
            )
        )
    depths = Counter(
        f"{r['role']}/{r['query_kind']}/{r['requested_depth']}"
        for r in accepted
        if r["requested_depth"] is not None
    )
    baseline = {}
    for mode in AnswerMode:
        train = [
            r
            for r in rows
            if r["split"] == "train-pilot" and r["contract"] == mode.value
        ]
        frequencies = Counter(r["answer"] for r in train)
        prediction = (
            min(frequencies, key=lambda x: (-frequencies[x], x))
            if frequencies
            else None
        )
        baseline[mode.value] = {
            "prediction_from_train_only": prediction,
            "roles": {
                role: {
                    "correct": sum(
                        r["answer"] == prediction
                        for r in rows
                        if r["split"] == role and r["contract"] == mode.value
                    ),
                    "total": sum(
                        r["split"] == role and r["contract"] == mode.value for r in rows
                    ),
                }
                for role in (*ROLES, "extrapolation-pilot")
            },
        }
    return {
        "attempted_bases": len(records),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "accepted_bases": len(accepted),
        "views_per_style": len(rows),
        "accepted_stories": len({r["story_id"] for r in accepted}),
        "accepted_families": len({r["family_id"] for r in accepted}),
        "accepted_story_keys": len({r["story_key"] for r in accepted}),
        "depth_base_counts": dict(sorted(depths.items())),
        "null_depth_reasons": dict(
            Counter(
                r["data"]["measurement"].get("reason")
                for r in accepted
                if r["data"]["measurement"]["depth"] is None
            )
        ),
        "tables": groups,
        "baseline": baseline,
        "token_maxima_before_admission": {
            style: {
                field: max(
                    (t[field] for t in all_tokens if t["style"] == style), default=0
                )
                for field in (
                    "target_tokens",
                    "training_tokens",
                    "stage2_reserved_tokens",
                )
            }
            for style in ("natural", "symbolic")
        },
    }


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite feasibility results")
    if args.tokenizer.name != REVISION:
        raise ValueError("expected the pinned Qwen tokenizer snapshot")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    solver = SpatialSolverV2(backend="z3", timeout_ms=5000)
    attempts = schedule()
    keys, bad_families = split_conflicts(attempts)
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "spatial/v2").glob("*.py"))
    protocol = (
        root
        / "experiments/spatialentail-composition-v1/research-analysis/DEPTH-FEASIBILITY-PROTOCOL.md"
    )
    hashes = {str(p.relative_to(root)): digest(p) for p in [*sources, protocol]}
    args.output.mkdir()
    manifest = {
        "status": "feasibility-only-not-production",
        "seed": SEED,
        "stories": len(attempts),
        "attempt_budget": len(attempts) * 3,
        "solver_timeout_ms": 5000,
        "source_sha256": hashes,
        "tokenizer_sha256": {
            p.name: digest(p) for p in sorted(args.tokenizer.iterdir()) if p.is_file()
        },
        "versions": {
            p: version(p) for p in ("z3-solver", "transformers", "tokenizers")
        },
        "schedule": [
            {
                "story_id": a.story.story_id,
                "family_id": a.story.source_family,
                "role": a.role,
                "requested_depth": a.requested_depth,
                "story_key": keys[a.story.story_id],
            }
            for a in attempts
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    with (args.output / "attempts.jsonl").open("w") as stream:
        for attempt in attempts:
            story = attempt.story
            for problem in story.problems():
                kind = type(problem.query).__name__.removesuffix("Query").lower()
                record = {
                    "story_id": story.story_id,
                    "family_id": story.source_family,
                    "story_key": keys[story.story_id],
                    "role": attempt.role,
                    "query_kind": kind,
                    "requested_depth": attempt.requested_depth,
                    "problem": tagged_dataclass_to_dict(problem),
                }
                try:
                    if story.source_family in bad_families:
                        record.update(
                            status="split-conflict",
                            reason="pre-admission family quarantine",
                        )
                    else:
                        data = build_base(attempt, problem, tokenizer, solver)
                        record["data"] = data
                        record["status"] = (
                            "accepted"
                            if all(t["admitted"] for t in data["tokens"])
                            else "token-rejected"
                        )
                except UnsupportedEvidence as exc:
                    record.update(status="unsupported", reason=str(exc))
                except (ValueError, RuntimeError) as exc:
                    record.update(
                        status="verification-failed",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                records.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
            print(
                f"{story.story_id}: {Counter(r['status'] for r in records)}", flush=True
            )
    if any(digest(root / p) != h for p, h in hashes.items()):
        raise ValueError(
            "source changed during feasibility run; result is not reproducible"
        )
    summary = summarize(records)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"tables", "depth_base_counts"}
            },
            indent=2,
        )
    )
    if summary["status_counts"].get("verification-failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
