"""Small unique-match Selection generator study; never a training release."""

import argparse
import json
import random
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from spatial.v2.composition_balance import build_attempt, query_profile
from spatial.v2.composition_feasibility import scramble
from spatial.v2.composition_pilot import Story, story_key
from spatial.v2.composition_selection import describe, inventory, select_milp
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.solver import And, Direction, SpatialSolverV2, direction_constraint

SEED = 20261031
PATTERNS = ("match-chain", "rival-chain", "shared-exclusion")


def make_story(pattern, depth, sample, rng, *, direction=None):
    if pattern not in PATTERNS or not 1 <= depth <= 10 or sample not in (0, 1):
        raise ValueError("outside fixed expansion scope")
    if direction is not None and not isinstance(direction, Direction):
        raise ValueError("direction must be a Direction")
    directions = tuple(Direction)
    diagonal = direction is not None and directions.index(direction) % 2 == 1
    match_direction = Direction.NORTHEAST if diagonal else Direction.NORTH
    candidates = tuple(f"C{i}" for i in range(rng.randint(5, 7)))
    objects = ["O", *candidates]
    atoms = []

    def add(subject, direction, reference):
        atoms.append(direction_constraint(subject, reference, direction))

    def chain(end, length, direction):
        previous = "O"
        for i in range(length - 1):
            node = f"Link{i}"
            objects.append(node)
            add(node, direction, previous)
            previous = node
        add(end, direction, previous)

    exclusions = (Direction.SOUTH, Direction.EAST, Direction.WEST)
    if pattern == "match-chain":
        chain("C0", depth, match_direction)
        for candidate in candidates[1:]:
            add(candidate, rng.choice(exclusions), "O")
    elif pattern == "rival-chain":
        add("C0", match_direction, "O")
        chain("C1", depth, Direction.SOUTH)
        for candidate in candidates[2:]:
            add(candidate, rng.choice(exclusions), "O")
    else:
        add("C0", match_direction, "O")
        previous = "O"
        for i in range(depth - 1):
            node = f"Trunk{i}"
            objects.append(node)
            add(node, Direction.SOUTH, previous)
            previous = node
        for candidate in candidates[1:]:
            add(candidate, rng.choice(exclusions), previous)
    # Only quarter-turns preserve the axis constraints. Diagonals start from
    # an actual NE construction, not a 45-degree relabeling of the N story.
    turn = (
        rng.randrange(4) * 2
        if direction is None
        else (directions.index(direction) - directions.index(match_direction)) % 8
    )
    atoms = [
        direction_constraint(
            atom.subject,
            atom.reference,
            directions[(directions.index(next(iter(atom.allowed))) + turn) % 8],
        )
        for atom in atoms
    ]
    story = Story(
        f"unique-expansion-{pattern}-d{depth}-s{sample}"
        + (f"-{direction.value}" if direction is not None else ""),
        f"unique-expansion-{pattern}-s{sample}",
        pattern,
        tuple(objects),
        And(tuple(atoms)),
        "C0",
        "O",
        candidates,
        directions[(directions.index(match_direction) + turn) % 8],
        (),
    )
    return scramble(story, rng)


def schedule():
    rng = random.Random(SEED)
    return tuple(
        (make_story(pattern, depth, sample, rng), depth)
        for pattern in PATTERNS
        for depth in range(1, 11)
        for sample in range(2)
    )


def deduplicate(old, additions):
    seen = {r["structure"]["key"] for r in old if r["status"] == "accepted"}
    kept, duplicates = [], []
    for record in sorted(additions, key=lambda r: r["base_id"]):
        if record["status"] != "accepted":
            continue
        key = record["structure"]["key"]
        if key in seen:
            duplicates.append(record["base_id"])
        else:
            kept.append(record)
            seen.add(key)
    return kept, duplicates


def summarize(additions, kept, duplicates, before, after, selection):
    accepted = [r for r in additions if r["status"] == "accepted"]
    tokens = [t for r in accepted for t in r["tokens"]]
    result = {
        "attempts": len(additions),
        "statuses": dict(Counter(r["status"] for r in additions)),
        "failures": [
            {k: r.get(k) for k in ("base_id", "stage", "status", "reason")}
            for r in additions
            if r["status"] != "accepted"
        ],
        "distinct_added_bases": len(kept),
        "duplicates_excluded": duplicates,
        "families": len({r["family_id"] for r in kept}),
        "depths": {
            pattern: {
                d: sum(
                    r["recipe"] == pattern and r["requested_depth"] == d for r in kept
                )
                for d in range(1, 11)
            }
            for pattern in PATTERNS
        },
        "accepted_views_per_style": sum(len(r["rows"]) for r in accepted),
        "tokens": {
            style: {
                "mean": sum(t["target_tokens"] for t in tokens if t["style"] == style)
                / max(1, sum(t["style"] == style for t in tokens)),
                "max": max(
                    (t["target_tokens"] for t in tokens if t["style"] == style),
                    default=0,
                ),
            }
            for style in ("natural", "symbolic")
        },
        "selection": selection,
        "before": before,
        "after": after,
    }
    return result


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.tokenizer.name != REVISION:
        raise ValueError("expected fresh output and pinned tokenizer")
    old_manifest = json.loads((args.source / "manifest.json").read_text())
    if any(
        digest(args.source / p) != h for p, h in old_manifest["output_sha256"].items()
    ):
        raise ValueError("old pool integrity mismatch")
    old = [
        json.loads(line)
        for line in (args.source / "attempts.jsonl").read_text().splitlines()
    ]
    previous = json.loads(args.comparison.read_text())
    if previous["selection"]["status"] != "optimal":
        raise ValueError("expected completed earlier comparison")
    root = Path(__file__).resolve().parents[2]
    protocol = (
        root
        / "experiments/spatialentail-composition-v1/research-analysis/UNIQUE-SELECTION-EXPANSION-PROTOCOL-v1.md"
    )
    sources = [
        *sorted((root / "spatial/v2").glob("*.py")),
        protocol,
        args.source / "attempts.jsonl",
        args.comparison,
    ]
    hashes = {str(p.resolve()): digest(p) for p in sources}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    solver = SpatialSolverV2(backend="z3", timeout_ms=5000)
    entries = schedule()
    args.output.mkdir()
    manifest = {
        "status": "running-development-only",
        "seed": SEED,
        "attempt_budget": 60,
        "source_sha256": hashes,
        "versions": {
            p: version(p)
            for p in ("scipy", "numpy", "z3-solver", "transformers", "tokenizers")
        },
        "tokenizer_sha256": {
            p.name: digest(p) for p in args.tokenizer.iterdir() if p.is_file()
        },
        "schedule": [
            {"story_id": s.story_id, "pattern": s.recipe, "depth": d}
            for s, d in entries
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    additions = []
    with (args.output / "attempts.jsonl").open("w") as stream:
        for story, depth in entries:
            problem = story.problems()[1]
            record = build_attempt(story, problem, depth, tokenizer, solver)
            record.update(
                story_key=story_key(story.objects, story.premise),
                structure=query_profile(problem),
            )
            if record["status"] == "accepted" and (
                not record["semantics"]["consistent"]
                or record["semantics"]["possible"] != [story.target]
                or record["semantics"]["certain"] != [story.target]
            ):
                record.update(
                    status="verification-failed",
                    stage="unique-match-check",
                    reason="did not establish exactly the nominated match",
                )
            additions.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(
                f"{len(additions)}/60: {dict(Counter(r['status'] for r in additions))}",
                flush=True,
            )
    kept, duplicates = deduplicate(old, additions)
    combined = inventory([*old, *kept])
    before = describe(combined, previous["selection"]["bundle_ids"])
    selection = select_milp(combined)
    after = (
        describe(combined, selection["bundle_ids"])
        if "bundle_ids" in selection
        else None
    )
    result = summarize(additions, kept, duplicates, before, after, selection)
    result["pool_unique_selection_before"] = sum(
        r["query_kind"] == "which"
        and any(
            row["contract"] == "single" and row["returned_count"] is not None
            for row in r.get("rows", [])
        )
        for r in old
        if r["status"] == "accepted"
    )
    result["pool_unique_selection_after"] = result[
        "pool_unique_selection_before"
    ] + len(kept)
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("input or source changed during expansion")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest["status"] = "completed-development-only"
    manifest["output_sha256"] = {
        p.name: digest(p) for p in args.output.iterdir() if p.name != "manifest.json"
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("before", "after", "selection")
            },
            indent=2,
        )
    )
    print("selection:", selection["status"])


if __name__ == "__main__":
    main()
