"""Bounded synthetic composition pilot. Never emits a training-ready release."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

from spatial.v2.composition_contracts import (
    UnsupportedEvidence,
    VerificationError,
    analyze,
    answer_domain,
    encode,
    parse_answer,
    parse_prompt,
    render_prompt,
)
from spatial.v2.composition_evidence import build_evidence, render_pair
from spatial.v2.difficulty import derive_axis_proof
from spatial.v2.grading import AnswerMode
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    SpatialProblem,
    SpatialSolverV2,
    WhichQuery,
    direction_constraint,
)
from spatial.v2.structure import canonical_structure_profile

RECIPES = (
    "chain",
    "fork",
    "empty",
    "singleton",
    "multiple",
    "contingent",
    "joint",
    "cycle",
    "coincidence",
)


@dataclass(frozen=True)
class Story:
    story_id: str
    source_family: str
    recipe: str
    objects: tuple[str, ...]
    premise: And
    target: str
    reference: str
    candidates: tuple[str, ...]
    direction: Direction
    distractors: tuple[int, ...]

    def problems(self):
        return (
            SpatialProblem(
                self.objects, self.premise, DirectionQuery(self.target, self.reference)
            ),
            SpatialProblem(
                self.objects,
                self.premise,
                WhichQuery(
                    frozenset({self.direction}), self.reference, self.candidates
                ),
            ),
            SpatialProblem(
                self.objects,
                self.premise,
                CountQuery(
                    frozenset({self.direction}), self.reference, self.candidates
                ),
            ),
        )


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def story_key(objects, premise):
    """Conservative rename/order/inverse invariant, independent of all queries."""
    sentinel = "StorySentinel"
    while sentinel in objects:
        sentinel += "X"
    directions = tuple(Direction)
    atoms = set()
    for atom in premise.operands:
        direction = next(iter(atom.allowed))
        atoms.add(atom)
        inverse = directions[(directions.index(direction) + 4) % 8]
        atoms.add(direction_constraint(atom.reference, atom.subject, inverse))
    projection = SpatialProblem(
        (*objects, sentinel),
        And(tuple(sorted(atoms, key=repr))),
        WhichQuery(frozenset({Direction.NORTH}), sentinel, objects),
    )
    profile = canonical_structure_profile(projection, max_permutations=720)
    return _digest(profile["canonical_problem"])


def construct_stories(seed=20261028, per_recipe=5):
    if not 1 <= per_recipe <= 5:
        raise ValueError(
            "pilot per_recipe must be between one and five; no automatic expansion"
        )
    rng = random.Random(seed)
    stories = []
    directions = tuple(Direction)
    for recipe in RECIPES:
        for sample in range(per_recipe):
            family = f"{recipe}-{sample}"
            count = (5 if recipe == "fork" else 4) + sample % 3
            candidates = tuple(f"C{i}" for i in range(count))
            atoms = []
            objects = ["O", *candidates]

            def add(a, direction, b, atoms=atoms):
                atoms.append(direction_constraint(a, b, direction))

            length = 2 + sample % 3
            if recipe == "chain":
                previous = "O"
                for i in range(length - 1):
                    node = f"Link{i}"
                    objects.append(node)
                    add(node, Direction.NORTH, previous)
                    previous = node
                add("C0", Direction.NORTHEAST, previous)
                for i in range(1, count):
                    add(f"C{i}", Direction.SOUTH, "O")
            elif recipe == "fork":
                for i in range(count):
                    add(f"C{i}", Direction.EAST, "O")
            elif recipe in {"empty", "singleton", "multiple"}:
                matches = {"empty": 0, "singleton": 1, "multiple": count}[recipe]
                for i in range(count):
                    if i < matches:
                        # Variable-length support, not copies of the fixture premises.
                        previous = "O"
                        for j in range(sample % 3):
                            node = f"Link{i}X{j}"
                            objects.append(node)
                            add(node, Direction.NORTH, previous)
                            previous = node
                        add(f"C{i}", Direction.NORTH, previous)
                    else:
                        add(
                            f"C{i}",
                            rng.choice(
                                (Direction.EAST, Direction.WEST, Direction.SOUTH)
                            ),
                            "O",
                        )
            elif recipe == "contingent":
                objects.append("Anchor")
                add("C0", Direction.NORTHEAST, "Anchor")
                add("O", Direction.NORTHEAST, "Anchor")
                for i in range(1, count):
                    add(f"C{i}", Direction.SOUTH, "O")
            elif recipe == "joint":
                for i in range(count - 1):
                    add(f"C{i}", Direction.EAST, f"C{i + 1}")
            elif recipe == "cycle":
                nodes = tuple(f"Cycle{i}" for i in range(length + 1))
                objects.extend(nodes)
                for a, b in pairwise(nodes):
                    add(a, Direction.EAST, b)
                add(nodes[0], Direction.EAST, nodes[-1])
                for i in range(count):
                    add(f"C{i}", Direction.NORTH, "O")
            else:
                objects.append("Anchor")
                add("C0", Direction.NORTH, "O")
                add("C1", Direction.NORTH, "O")
                add("C0", Direction.EAST, "Anchor")
                add("C1", Direction.EAST, "Anchor")
                for i in range(2, count):
                    add(f"C{i}", Direction.SOUTH, "O")
            variants = [("base", tuple(atoms))]
            if recipe == "cycle":
                dirty = list(atoms)
                closing = length
                edge = dirty[closing]
                dirty[closing] = direction_constraint(
                    edge.reference, edge.subject, Direction.EAST
                )
                variants.append(("inconsistent", tuple(dirty)))
            turn = rng.randrange(4) * 2
            for variant, source in variants:
                transformed = []
                for atom in source:
                    direction = directions[
                        (directions.index(next(iter(atom.allowed))) + turn) % 8
                    ]
                    a, b = atom.subject, atom.reference
                    if rng.random() < 0.5:
                        a, b = b, a
                        direction = directions[(directions.index(direction) + 4) % 8]
                    transformed.append(direction_constraint(a, b, direction))
                extra_objects = list(objects)
                marked = []
                for i in range(sample % 3):
                    a, b = f"Extra{i}A", f"Extra{i}B"
                    extra_objects.extend((a, b))
                    atom = direction_constraint(a, b, rng.choice(directions))
                    transformed.append(atom)
                    marked.append(atom)
                rng.shuffle(transformed)
                target, reference = ("C0", "C1") if recipe == "fork" else ("C0", "O")
                stories.append(
                    Story(
                        f"{family}-{variant}",
                        family,
                        recipe,
                        tuple(extra_objects),
                        And(tuple(transformed)),
                        target,
                        reference,
                        tuple(c for c in candidates if c != reference),
                        directions[turn],
                        tuple(
                            i for i, atom in enumerate(transformed) if atom in marked
                        ),
                    )
                )
    return tuple(stories)


def assign_splits(stories, seed, *, stratified=False):
    """Union source families sharing any canonical story before assigning splits."""
    parent = {story.source_family: story.source_family for story in stories}

    def root(value):
        while parent[value] != value:
            value = parent[value]
        return value

    seen = {}
    for story in stories:
        key = story_key(story.objects, story.premise)
        if key in seen:
            a, b = root(story.source_family), root(seen[key])
            parent[max(a, b)] = min(a, b)
        seen[key] = story.source_family
    groups = defaultdict(list)
    for source in parent:
        groups[root(source)].append(source)
    families = {
        source: "family-" + _digest(sorted(members))[:16]
        for members in groups.values()
        for source in members
    }
    strata = defaultdict(set)
    recipes = defaultdict(set)
    for story in stories:
        recipes[families[story.source_family]].add(story.recipe)
    for family in families.values():
        key = tuple(sorted(recipes[family])) if stratified else ()
        strata[key].add(family)
    split = {}
    for members in strata.values():
        ordered = sorted(members, key=lambda f: _digest([seed, f]))
        # Source-only stratification precedes queries and solver admission.
        n = len(ordered)
        n_test, n_dev = max(1, n // 5), max(1, n // 5)
        if stratified and n < 3:
            n_test = n_dev = 0
        split.update(
            {
                family: "test"
                if i < n_test
                else "dev"
                if i < n_test + n_dev
                else "train"
                for i, family in enumerate(ordered)
            }
        )
    return {
        s.story_id: (families[s.source_family], split[families[s.source_family]])
        for s in stories
    }


def menus_for(semantics, seed):
    rng = random.Random(seed)
    domain = list(answer_domain(semantics.problem))
    if len(domain) < 4:
        raise UnsupportedEvidence("answer domain has fewer than four real values")
    possible = [v for v in domain if v in semantics.possible]
    fillers = [v for v in domain if v not in semantics.possible]
    rng.shuffle(possible)
    rng.shuffle(fillers)
    menu = (possible + fillers)[:4]
    rng.shuffle(menu)
    menus = [tuple(menu)]
    # Change only one slot so shared values retain their positions.
    if possible and len(possible) <= 4:
        omitted = next(v for v in menu if v in possible)
        replacement = next((v for v in fillers if v not in menu), None)
        if replacement is not None:
            partial = list(menu)
            partial[partial.index(omitted)] = replacement
            menus.append(tuple(partial))
    return tuple(menus)


def renamed_problem(problem):
    names = {obj: f"Renamed{i}" for i, obj in enumerate(reversed(problem.objects))}
    query = problem.query
    if isinstance(query, DirectionQuery):
        query = replace(
            query, target=names[query.target], reference=names[query.reference]
        )
    else:
        query = replace(
            query,
            reference=names[query.reference],
            candidates=tuple(names[c] for c in query.candidates),
        )
    result = replace(
        problem,
        objects=tuple(names[obj] for obj in problem.objects),
        query=query,
        premise=And(
            tuple(
                replace(
                    atom, subject=names[atom.subject], reference=names[atom.reference]
                )
                for atom in reversed(problem.premise.operands)
            )
        ),
    )
    return result, names


def _signature(semantics):
    return semantics.consistent, semantics.possible, semantics.certain


def verify_base(story, problem, semantics):
    variant, names = renamed_problem(problem)
    changed = analyze(variant)
    expected = _signature(semantics)
    if isinstance(problem.query, WhichQuery):
        expected = (
            semantics.consistent,
            frozenset(names[v] for v in semantics.possible),
            frozenset(names[v] for v in semantics.certain),
        )
    if _signature(changed) != expected:
        raise VerificationError("rename/order variant changed semantics")
    if story_key(problem.objects, problem.premise) != story_key(
        variant.objects, variant.premise
    ):
        raise VerificationError("canonical story grouping is not rename invariant")
    removals = [(i,) for i in story.distractors]
    if story.distractors:
        removals.append(story.distractors)
    for removed in removals:
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
        if _signature(analyze(reduced)) != _signature(semantics):
            raise VerificationError("proposed distractors alter complete semantics")
    chain = None
    query = problem.query
    if (
        isinstance(query, DirectionQuery)
        and semantics.consistent
        and len(semantics.possible) == 1
    ):
        proof = derive_axis_proof(
            problem,
            query.target,
            query.reference,
            Direction(next(iter(semantics.possible))),
        )
        if proof:
            chain = {
                axis: len(getattr(proof, axis).premise_indices) for axis in ("x", "y")
            }
    reference = "not-run: limited to Direction with at most six objects"
    if isinstance(query, DirectionQuery) and len(problem.objects) <= 6:
        other = analyze(problem, solver=SpatialSolverV2(backend="reference"))
        if _signature(other) != _signature(semantics):
            raise VerificationError("reference backend disagreement")
        reference = "passed: shared domain types; independent reference search"
    return {
        "rename_order": "passed",
        "verified_distractor_count": len(story.distractors),
        "distractor_premise_ids": [f"p{i + 1}" for i in story.distractors],
        "supporting_axis_path_lengths": chain,
        "reference_crosscheck": reference,
    }


def check_splits(rows, bases):
    groups = defaultdict(set)
    for row in rows:
        base = bases[row["base_id"]]
        for key in (row["family_id"], row["story_id"], base["story_key"]):
            groups[key].add(row["split"])
        if row["split"] != base["split"] or row["family_id"] != base["family_id"]:
            raise VerificationError("row and base split assignment disagree")
    if any(len(splits) != 1 for splits in groups.values()):
        raise VerificationError("story/family leakage across splits")


def run_pilot(
    seed=20261028, per_recipe=5, *, stories=None, compact=False, stratified=False
):
    stories = construct_stories(seed, per_recipe) if stories is None else stories
    assignments = assign_splits(stories, seed, stratified=stratified)
    rows, bases, attempts = [], {}, []
    for story in stories:
        family, split = assignments[story.story_id]
        for problem in story.problems():
            kind = type(problem.query).__name__.removesuffix("Query").lower()
            base_id = f"{story.story_id}-{kind}"
            attempt = {
                "base_id": base_id,
                "family_id": family,
                "split": split,
                "recipe": story.recipe,
                "query_kind": kind,
            }
            try:
                evidence = build_evidence(problem)
                semantics = evidence.semantics
                checks = verify_base(story, problem, semantics)
                base_rows = []
                for menu_index, menu in enumerate(
                    menus_for(semantics, _digest([seed, base_id]))
                ):
                    for mode in AnswerMode:
                        decision = encode(semantics, menu, mode)
                        prompt = render_prompt(problem, menu, mode)
                        if parse_prompt(prompt) != (problem, menu, mode):
                            raise VerificationError("prompt round trip mismatch")
                        permuted = tuple(reversed(menu))
                        other = encode(semantics, permuted, mode)
                        selected = {
                            menu["ABCD".index(c)]
                            for c in decision.letters
                            if c in "ABCD"
                        }
                        selected_other = {
                            permuted["ABCD".index(c)]
                            for c in other.letters
                            if c in "ABCD"
                        }
                        if selected != selected_other or (
                            decision.letters in {"E", "F"}
                            and other.letters != decision.letters
                        ):
                            raise VerificationError(
                                "menu permutation changed semantic selection"
                            )
                        if parse_prompt(render_prompt(problem, permuted, mode)) != (
                            problem,
                            permuted,
                            mode,
                        ):
                            raise VerificationError(
                                "permuted prompt round trip mismatch"
                            )
                        pair = render_pair(
                            evidence, menu, mode, compact_unique_direction=compact
                        )
                        if any(
                            parse_answer(text, mode, stage=1) != decision.letters
                            for text in pair.values()
                        ):
                            raise VerificationError("reference answer mismatch")
                        base_rows.append(
                            {
                                "id": f"{base_id}-m{menu_index}-{mode.value}",
                                "base_id": base_id,
                                "story_id": story.story_id,
                                "family_id": family,
                                "split": split,
                                "recipe": story.recipe,
                                "query_kind": kind,
                                "contract": mode.value,
                                "semantic_case": semantics.case,
                                "menu": menu,
                                "coverage": "full"
                                if semantics.possible <= set(menu)
                                else "partial",
                                "answer": decision.letters,
                                "escape_reason": decision.escape_reason,
                                "prompt": prompt,
                                "references": pair,
                            }
                        )
                bases[base_id] = {
                    "story_key": story_key(story.objects, story.premise),
                    "family_id": family,
                    "split": split,
                    "checks": checks,
                    "evidence": tagged_dataclass_to_dict(evidence),
                }
                rows.extend(base_rows)
                attempt.update(status="accepted", rows=len(base_rows))
            except UnsupportedEvidence as exc:
                attempt.update(
                    status="unsupported",
                    reason=str(exc),
                    problem=tagged_dataclass_to_dict(problem),
                )
            except (ValueError, RuntimeError) as exc:
                attempt.update(
                    status="verification-failed",
                    reason=f"{type(exc).__name__}: {exc}",
                    problem=tagged_dataclass_to_dict(problem),
                )
            attempts.append(attempt)
    check_splits(rows, bases)
    coverage = Counter(
        f"{r['query_kind']} / {r['contract']} / {r['semantic_case']}" for r in rows
    )
    samples = {}
    for row in rows:
        cell = f"{row['query_kind']} / {row['contract']} / {row['semantic_case']}"
        samples.setdefault(cell, row["id"])
    category_bases = defaultdict(list)
    for row in rows:
        if row["base_id"] not in category_bases[row["recipe"]]:
            category_bases[row["recipe"]].append(row["base_id"])
    sample_ids = set(samples.values())
    for ids in category_bases.values():
        distinct_stories = {}
        for base in ids:
            row = next(row for row in rows if row["base_id"] == base)
            distinct_stories.setdefault(row["story_id"], base)
        for base in list(distinct_stories.values())[:5]:
            sample_ids.add(next(row["id"] for row in rows if row["base_id"] == base))
    return {
        "status": "pilot-only-not-training-ready",
        "config": {"seed": seed, "per_recipe": per_recipe, "recipes": RECIPES},
        "split_assignments": assignments,
        "counts": {
            "attempted_stories": len(stories),
            "accepted_stories": len({r["story_id"] for r in rows}),
            "canonical_story_groups": len({b["story_key"] for b in bases.values()}),
            "attempted_bases": len(attempts),
            "accepted_bases": len(bases),
            "views": len(rows),
            "families": len({value[0] for value in assignments.values()}),
            "attempt_status": dict(Counter(a["status"] for a in attempts)),
            "splits": dict(Counter(r["split"] for r in rows)),
            "answer_frequencies": dict(Counter(r["answer"] for r in rows)),
            "reference_crosschecks": dict(
                Counter(b["checks"]["reference_crosscheck"] for b in bases.values())
            ),
            "coverage_frequencies": dict(Counter(r["coverage"] for r in rows)),
        },
        "coverage": dict(sorted(coverage.items())),
        "coverage_by_split": {
            split: dict(
                sorted(
                    Counter(
                        f"{r['query_kind']} / {r['contract']} / {r['semantic_case']}"
                        for r in rows
                        if r["split"] == split
                    ).items()
                )
            )
            for split in ("train", "dev", "test")
        },
        "attempts": attempts,
        "bases": bases,
        "rows": rows,
        "audit": {
            "status": "pending",
            "sample_ids": sorted(sample_ids),
            "cell_samples": samples,
        },
        "remaining_gates": [
            "Manual audit must be recorded before any release; recipe is not a frozen evidence taxonomy.",
            "Reference crosschecks cover only Direction with <=6 objects and share domain/sign definitions; Selection/Count and larger cases are not crosschecked.",
            "Final evidence-category matrix, sampling, tokenizer measurement and runtime decoding are not frozen.",
            "Templates are bounded constructions, not representative sampling of all atomic maps.",
            "No training release or automatic dataset expansion is authorized.",
        ],
    }


def write_pilot(pack, output):
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "pilot.json": json.dumps(pack, indent=2) + "\n",
        "model-inputs.jsonl": "".join(
            json.dumps(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "messages": [{"role": "user", "content": row["prompt"]}],
                }
            )
            + "\n"
            for row in pack["rows"]
        ),
    }
    report = [
        "# Composition Generator Pilot",
        "",
        f"Status: {pack['status']}. Split assignment precedes evidence acceptance.",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(pack["counts"], indent=2),
        "```",
        "",
        "## Coverage",
        "",
        "| Query / Contract / Case | Views |",
        "|---|---:|",
        *(f"| {cell} | {count} |" for cell, count in pack["coverage"].items()),
        "",
        "## Split Coverage Gaps",
        "",
        *(
            f"- {split}: missing {', '.join(sorted(set(pack['coverage']) - set(cells))) or 'none'}."
            for split, cells in pack["coverage_by_split"].items()
        ),
        "",
        "## Rejections",
        "",
        *(
            f"- {a['base_id']} ({a['split']}): {a['status']}: {a.get('reason', '')}"
            for a in pack["attempts"]
            if a["status"] != "accepted"
        ),
        *(
            []
            if any(a["status"] != "accepted" for a in pack["attempts"])
            else [
                "None in this bounded run. This does not establish support for arbitrary atomic stories."
            ]
        ),
        "",
        "## Remaining Gates",
        "",
        *(f"- {gate}" for gate in pack["remaining_gates"]),
        "",
        "Manual audit status: pending. See audit-samples.md for selected paired examples.",
    ]
    payloads["report.md"] = "\n".join(report) + "\n"
    audit = [
        "# Pilot Audit Samples",
        "",
        "Review status: pending. Inspect both styles and record outcomes.",
        "",
    ]
    for row in pack["rows"]:
        if row["id"] not in pack["audit"]["sample_ids"]:
            continue
        audit.extend(
            [
                f"## {row['id']}",
                f"Split: {row['split']}; case: {row['semantic_case']}; answer: {row['answer']}",
                "",
                "### Prompt",
                "```text",
                row["prompt"],
                "```",
                "### Natural",
                "```text",
                row["references"]["natural"],
                "```",
                "### Symbolic",
                "```text",
                row["references"]["symbolic"],
                "```",
                "",
            ]
        )
    payloads["audit-samples.md"] = "\n".join(audit)
    for name, content in payloads.items():
        (output / name).write_text(content)
    root = Path(__file__).resolve().parents[2]
    # Training/export utilities do not participate in generation.
    sources = sorted(
        path
        for path in (root / "spatial/v2").glob("*.py")
        if path.name not in {"composition_training.py", "composition_evaluation.py"}
    )
    receipt = {
        "status": pack["status"],
        "solver_version": version("z3-solver"),
        "config": pack["config"],
        "manual_audit": "pending",
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "output_sha256": {
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in payloads.items()
        },
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--per-recipe", type=int, default=5)
    args = parser.parse_args()
    pack = run_pilot(args.seed, args.per_recipe)
    write_pilot(pack, args.output)
    print(json.dumps(pack["counts"], indent=2))
    if any(a["status"] == "verification-failed" for a in pack["attempts"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
