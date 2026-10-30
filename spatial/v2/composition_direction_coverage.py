"""Fixed all-direction Selection coverage check, not production sampling."""

import argparse
import json
import random
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from spatial.v2.composition_balance import build_attempt, menu_inventory
from spatial.v2.composition_depth import measure_depth
from spatial.v2.composition_feasibility import stable_int
from spatial.v2.composition_selection_expansion import PATTERNS, make_story
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import Direction, SpatialSolverV2

SEED = 20261101


def schedule():
    # Rotations deliberately reuse construction choices; they are not new families.
    return tuple(
        (
            make_story(
                pattern,
                depth,
                0,
                random.Random(stable_int(f"{SEED}/{pattern}/{depth}")),
                direction=direction,
            ),
            depth,
        )
        for pattern in PATTERNS
        for depth in range(1, 11)
        for direction in Direction
    )


def check_target(record, story, depth):
    if record["status"] != "accepted":
        return
    semantics = record["semantics"]
    memberships = record["measurement"]["memberships"]
    positive = [m["depth"] for m in memberships if m["qualifies"]]
    negative = [m["depth"] for m in memberships if not m["qualifies"]]
    valid = (
        semantics["consistent"]
        and semantics["possible"] == semantics["certain"] == [story.target]
        and record["measurement"]["depth"] == depth
        and record["measurement"]["dependence_check"] == "passed-not-a-minimality-proof"
        and any(a["answer_changed"] for a in record["measurement"]["ablations"])
        and positive == [depth if story.recipe == "match-chain" else 1]
        and max(negative) == (1 if story.recipe == "match-chain" else depth)
        and (story.recipe != "shared-exclusion" or all(d == depth for d in negative))
        and {
            row["contract"]
            for row in record["rows"]
            if row["substantive_depth_support"]
        }
        == {"single", "visible-possible", "all-possible"}
    )
    if not valid:
        record.update(
            status="verification-failed",
            stage="target-coverage",
            reason="unique/depth support mismatch",
        )


def depth_visible_menus(semantics, seed):
    """Keep a deepest exclusion visible when its match is displayed."""
    menus = menu_inventory(semantics, seed)
    measured = measure_depth(semantics.problem)
    deep_rivals = {
        m["candidate"]
        for m in measured["memberships"]
        if not m["qualifies"] and m["depth"] == measured["depth"]
    }
    if not deep_rivals:
        return menus
    revised = []
    for original in menus:
        menu = list(original)
        if semantics.possible.intersection(menu) and not deep_rivals.intersection(menu):
            index = next(
                i
                for i, candidate in enumerate(menu)
                if candidate not in semantics.possible
            )
            menu[index] = min(deep_rivals)
        revised.append(tuple(menu))
    return tuple(revised)


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        help="Recheck only failed records from an earlier fixed run",
    )
    args = parser.parse_args()
    if args.output.exists() or args.tokenizer.name != REVISION:
        raise ValueError("expected fresh output and pinned tokenizer")
    root = Path(__file__).resolve().parents[2]
    protocol = (
        root
        / "experiments/spatialentail-composition-v1/research-analysis"
        / (
            "DIRECTION-COVERAGE-PROTOCOL-v2.md"
            if args.source
            else "DIRECTION-COVERAGE-PROTOCOL-v1.md"
        )
    )
    hashes = {
        str(p.resolve()): digest(p)
        for p in [*sorted((root / "spatial/v2").glob("*.py")), protocol]
    }
    entries = schedule()
    prior = {}
    if args.source:
        source_manifest = json.loads((args.source / "manifest.json").read_text())
        for name, expected in source_manifest["output_sha256"].items():
            if digest(args.source / name) != expected:
                raise ValueError("source output hash mismatch")
        prior = {
            r["story_id"]: r
            for r in (
                json.loads(line)
                for line in (args.source / "attempts.jsonl").read_text().splitlines()
            )
        }
        if set(prior) != {s.story_id for s, _ in entries}:
            raise ValueError("source does not match the fixed schedule")
        for path in args.source.iterdir():
            if path.is_file():
                hashes[str(path.resolve())] = digest(path)
    manifest = {
        "status": "running-development-only",
        "seed": SEED,
        "attempt_budget": len(entries),
        "source_sha256": hashes,
        "versions": {
            p: version(p) for p in ("z3-solver", "transformers", "tokenizers")
        },
        "tokenizer_sha256": {
            p.name: digest(p) for p in args.tokenizer.iterdir() if p.is_file()
        },
        "schedule": [
            {
                "story_id": s.story_id,
                "family": s.source_family,
                "pattern": s.recipe,
                "direction": s.direction.value,
                "depth": d,
            }
            for s, d in entries
        ],
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    solver = SpatialSolverV2(backend="z3", timeout_ms=5000)
    args.output.mkdir()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records, examples, rebuilt = [], [], []
    with (args.output / "attempts.jsonl").open("w") as stream:
        for story, depth in entries:
            problem = story.problems()[1]
            old = prior.get(story.story_id)
            if old and old["problem"] != tagged_dataclass_to_dict(problem):
                raise ValueError("source problem changed")
            if old and old["status"] == "accepted":
                record = old
            else:
                record = build_attempt(
                    story,
                    problem,
                    depth,
                    tokenizer,
                    solver,
                    menu_selector=depth_visible_menus
                    if args.source
                    else menu_inventory,
                )
                rebuilt.append(record["base_id"])
                if old:
                    record["previous_failure"] = {
                        k: old.get(k) for k in ("stage", "reason")
                    }
            record["requested_direction"] = story.direction.value
            check_target(record, story, depth)
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(
                f"{len(records)}/{len(entries)}: {dict(Counter(r['status'] for r in records))}",
                flush=True,
            )
            if (
                record["status"] == "accepted"
                and depth in (3, 10)
                and story.direction is Direction.NORTHEAST
            ):
                row = next(
                    r
                    for r in record["rows"]
                    if r["contract"] == "single" and r["substantive_depth_support"]
                )
                examples.append(
                    f"## {story.recipe}, Northeast, depth {depth}\n\n`{row['id']}`\n\n"
                    f"### Prompt\n\n```text\n{row['prompt']}\n```\n\n"
                    f"### Natural\n\n{row['references']['natural']}\n\n"
                    f"### Symbolic\n\n```text\n{row['references']['symbolic']}\n```\n"
                )
    tokens = [t for r in records if r["status"] == "accepted" for t in r["tokens"]]
    result = {
        "attempts": len(records),
        "rebuilt_bases": rebuilt,
        "inherited_verified_bases": len(records) - len(rebuilt),
        "statuses": dict(Counter(r["status"] for r in records)),
        "failures": [
            {k: r.get(k) for k in ("base_id", "stage", "status", "reason")}
            for r in records
            if r["status"] != "accepted"
        ],
        "coverage": [
            {
                "pattern": r["recipe"],
                "direction": r["requested_direction"],
                "depth": r["requested_depth"],
                "status": r["status"],
            }
            for r in records
        ],
        "families": len({r["family_id"] for r in records}),
        "accepted_views_per_style": sum(
            len(r["rows"]) for r in records if r["status"] == "accepted"
        ),
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
    }
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("source changed during coverage check")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "examples.md").write_text(
        "# Diagonal Selection Examples\n\nDevelopment-only; rotations are not independent construction families.\n\n"
        + "\n".join(examples)
    )
    manifest["status"] = "completed-development-only"
    manifest["all_checks_passed"] = not result["failures"]
    manifest["output_sha256"] = {
        p.name: digest(p) for p in args.output.iterdir() if p.name != "manifest.json"
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("completed:", manifest["all_checks_passed"])


if __name__ == "__main__":
    main()
