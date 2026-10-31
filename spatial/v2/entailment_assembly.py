"""Assemble a versioned candidate from integrity-checked saved admissions."""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from spatial.v2.composition_execution import (
    Checkpoints,
    atomic_json,
    fingerprint,
    work_identity,
    work_items,
    worker_admit,
)
from spatial.v2.composition_execution import worker_init as admission_init
from spatial.v2.composition_production import (
    preflight,
    source_lineage,
    stratum,
    summary_record,
    write_exports,
)
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.entailment_positions import VERSION as POSITION_VERSION
from spatial.v2.entailment_positions import balanced_metadata, render_record
from spatial.v2.entailment_production import has_shortfalls, schedule, summarize
from spatial.v2.entailment_retention import (
    matches,
    quota_keys,
    quota_targets,
    requirements,
    select,
)

_tokenizer = None


def revision_identity(config, tokenizer):
    return {
        "schema": "entailment-saved-pool-assembly-v1",
        "config_sha256": digest(config),
        "menu_position_policy": POSITION_VERSION,
        "sources_sha256": {
            path.name: digest(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "tokenizer_sha256": {
            path.name: digest(path)
            for path in sorted(tokenizer.iterdir())
            if path.is_file()
        },
        "versions": {
            package: version(package)
            for package in (
                "z3-solver",
                "transformers",
                "tokenizers",
                "scipy",
                "jinja2",
                "PyYAML",
                "numpy",
            )
        },
    }


def check_origin(source, config, tokenizer_hashes):
    identity = json.loads((source / "manifest.json").read_text())["identity"]
    if identity["config_sha256"] != digest(config):
        raise ValueError("assembly configuration differs from the saved execution")
    if identity["policy"] != json.loads(config.read_text()):
        raise ValueError("saved policy differs from the supplied configuration")
    if identity["tokenizer_sha256"] != tokenizer_hashes:
        raise ValueError("assembly tokenizer differs from the saved execution")


def worker_init(path):
    global _tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)


def render(task):
    original, metadata = task
    return render_record(original, metadata, _tokenizer)


def saved_candidates(source):
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] not in (
        "interrupted",
        "failed",
        "completed-unsealed-candidate",
    ):
        raise ValueError("stop the source execution before assembling a revision")
    journal = source / "attempts.jsonl.tmp"
    if not journal.exists():
        journal = source / "attempts.jsonl"
    records = {}
    with journal.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            record = json.loads(line)
            if record["status"] == "accepted":
                if record["base_id"] in records:
                    raise ValueError("duplicate saved base")
                records[record["base_id"]] = record
    verified, hashes, task_hashes = set(), {}, {}
    frontier = -1
    identity = fingerprint(manifest["identity"])
    for path in sorted((source / "checkpoints").glob("*.json")):
        envelope = json.loads(path.read_text())
        payload = envelope["payload"]
        if (
            fingerprint(payload) != envelope["sha256"]
            or payload["identity"] != identity
            or payload["index"] != int(path.stem)
        ):
            raise ValueError(f"saved checkpoint integrity failure: {path.name}")
        base = payload["record"]["base_id"]
        task_hashes[str(payload["index"])] = payload.get("task")
        if base in records:
            if any(records[base].get(k) != v for k, v in payload["record"].items()):
                raise ValueError("journal differs from immutable admission checkpoint")
            verified.add(base)
            hashes[path.name] = digest(path)
            frontier = max(frontier, payload["index"])
    if verified != set(records):
        raise ValueError("saved journal has an admission without a verified checkpoint")
    return list(records.values()), {
        "source": str(source),
        "manifest_sha256": digest(source / "manifest.json"),
        "journal_sha256": digest(journal),
        "checkpoint_hashes": hashes,
        "completed_task_hashes": task_hashes,
        "consumed_frontier": frontier,
    }


def supply_ready(candidates, policy):
    counts = Counter(key for record in candidates for key in quota_keys(record))
    if any(counts[key] < n for key, n in quota_targets(policy).items()):
        return False
    for rule in requirements(policy):
        found = [record for record in candidates if matches(record, rule)]
        if len(found) < rule.get("minimum_bases", 0):
            return False
        if len({source_lineage(r["family_id"]) for r in found}) < rule.get(
            "minimum_families", 0
        ):
            return False
    return True


def retention_metadata(record, entry, story_key):
    """Add coordinator metadata to a raw admission without changing its evidence."""
    record = deepcopy(record)
    kind = record["query_kind"]
    record.update(
        story_key=story_key,
        stratum=stratum(entry, kind),
        source_stratum=f"{entry.split}/{kind}/{record['source']}",
    )
    if record["status"] == "accepted" and kind == "count" and entry.depth is not None:
        values = record["semantics"]["entailed"]
        if len(values) != 1 or type(values[0]) is not int:
            raise ValueError("controlled Count must have one entailed integer")
        record["controlled_count_value"] = values[0]
    return record


def complete_reserve(args, policy, origin, originals, compact, identity):
    """Continue only unused slots in the original ordered reserve, never refill."""
    print(json.dumps({"stage": "reconstructing-unused-fixed-reserve"}), flush=True)
    entries = schedule(policy)
    exclusion = args.config.parent / policy["development_exclusions"]["path"]
    if digest(exclusion) != policy["development_exclusions"]["sha256"]:
        raise ValueError("development exclusion inventory changed")
    keys = preflight(entries, json.loads(exclusion.read_text()))
    jobs = work_items(entries, policy)
    jobs_by_index = {task[0]: task for task in jobs}
    for index, saved in origin["completed_task_hashes"].items():
        if saved != fingerprint(work_identity(jobs_by_index[int(index)])):
            raise ValueError("saved checkpoint task differs from fixed reserve")
    remaining = [task for task in jobs if task[0] > origin["consumed_frontier"]]
    old = Checkpoints(
        args.source_run / "checkpoints",
        fingerprint(
            json.loads((args.source_run / "manifest.json").read_text())["identity"]
        ),
        jobs,
    )
    directory = args.output / "reserve-checkpoints"
    directory.mkdir()
    new = Checkpoints(directory, fingerprint(identity), jobs)
    reused, reused_indices = None, set()
    if getattr(args, "reserve_reuse", None) is not None:
        manifest_path = args.reserve_reuse / "manifest.json"
        prior = json.loads(manifest_path.read_text())
        prior_identity = prior["identity"]
        if (
            prior["status"] != "failed-unsealed-revision"
            or prior_identity["config_sha256"] != identity["config_sha256"]
            or prior_identity["tokenizer_sha256"] != identity["tokenizer_sha256"]
            or any(
                sha != identity["sources_sha256"].get(name)
                for name, sha in prior_identity["sources_sha256"].items()
                if name != Path(__file__).name
            )
        ):
            raise ValueError("reserve reuse differs beyond assembly implementation")
        reused = Checkpoints(
            args.reserve_reuse / "reserve-checkpoints",
            fingerprint(prior_identity),
            jobs,
        )
        reused_indices = reused.existing()
        origin["reserve_reuse"] = {
            "source": str(args.reserve_reuse),
            "manifest_sha256": digest(manifest_path),
            "checkpoint_sha256": {
                reused.path(i).name: digest(reused.path(i)) for i in reused_indices
            },
        }
    missing = [
        task
        for task in remaining
        if str(task[0]) not in origin["completed_task_hashes"]
        and task[0] not in reused_indices
    ]
    accepted_ids = {record["base_id"] for record in originals}
    statuses = Counter()
    with (
        multiprocessing.get_context("spawn").Pool(
            args.workers,
            initializer=admission_init,
            initargs=(policy, str(args.tokenizer)),
        ) as pool,
        (args.output / "reserve-attempts.jsonl").open("x") as stream,
    ):
        results = iter(pool.imap(worker_admit, missing))
        for number, (index, entry, _) in enumerate(remaining, 1):
            if str(index) in origin["completed_task_hashes"]:
                record = old.load(index)
            elif index in reused_indices:
                record = reused.load(index)
            else:
                actual, record, seconds, rss = next(results)
                if actual != index:
                    raise ValueError("reserve admission order changed")
                new.save(index, record, seconds, rss)
            record = retention_metadata(record, entry, keys[entry.story.story_id])
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            statuses[record["status"]] += 1
            if record["status"] == "verification-failed":
                raise ValueError("unused reserve produced a verification failure")
            if record["status"] == "accepted" and record["base_id"] not in accepted_ids:
                accepted_ids.add(record["base_id"])
                originals.append(record)
                compact.append(balanced_metadata(summary_record(record)))
            if number % 100 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "unused-fixed-reserve",
                            "consumed": number,
                            "remaining_slots": len(remaining),
                            "statuses": dict(statuses),
                        }
                    ),
                    flush=True,
                )
            if (
                record["status"] == "accepted"
                and (number % 100 == 0 or number == len(remaining))
                and supply_ready(compact, policy)
            ):
                selected = select(compact, policy)
                if selected is not None:
                    origin["reserve_completion"] = {
                        "consumed_slots": number,
                        "last_index": index,
                        "statuses": dict(statuses),
                        "new_seeds": False,
                        "order": "original fixed reserve",
                    }
                    return selected
    raise ValueError(
        "complete fixed reserve cannot satisfy the unchanged coverage gates"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reserve-reuse", type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh revision directory")
    if args.tokenizer.name != REVISION or not args.tokenizer.is_dir():
        parser.error("provide the pinned local tokenizer")
    if not 1 <= args.workers <= 128:
        parser.error("workers must be in 1-128")
    try:
        assemble(args)
    except BaseException as error:
        manifest_path = args.output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest.update(
                status="failed-unsealed-revision",
                acceptance=None,
                error=f"{type(error).__name__}: {error}",
            )
            atomic_json(manifest_path, manifest)
        raise


def assemble(args):
    identity = revision_identity(args.config, args.tokenizer)
    check_origin(args.source_run, args.config, identity["tokenizer_sha256"])
    originals, origin = saved_candidates(args.source_run)
    policy = json.loads(args.config.read_text())
    revised_policy = deepcopy(policy)
    revised_policy["menu_position_policy"] = POSITION_VERSION
    revised_policy["frozen_sources_sha256"] = identity["sources_sha256"]
    revised_policy["assembly_revision"] = {
        "schema": identity["schema"],
        "origin_config_sha256": identity["config_sha256"],
        "changes": ["correlated Selection menu positions", "Count singular wording"],
        "base_targets_changed": False,
        "coverage_minima_changed": False,
    }
    compact = [balanced_metadata(summary_record(record)) for record in originals]
    print(
        json.dumps(
            {"saved_verified_candidates": len(compact), "stage": "coverage-retention"}
        ),
        flush=True,
    )
    selected = select(compact, revised_policy)
    if selected is None:
        args.output.mkdir()
        atomic_json(
            args.output / "manifest.json",
            {
                "status": "completing-unused-fixed-reserve",
                "origin": origin,
                "identity": identity,
                "sealed": False,
                "training_authorized": False,
            },
        )
        selected = complete_reserve(
            args, revised_policy, origin, originals, compact, identity
        )
    tasks = [
        (original, metadata)
        for original, metadata in zip(originals, compact)
        if original["base_id"] in selected
    ]
    if not args.output.exists():
        args.output.mkdir()
    atomic_json(
        args.output / "manifest.json",
        {
            "status": "rendering-unsealed-reference-revision",
            "origin": origin,
            "identity": identity,
            "training_authorized": False,
            "sealed": False,
        },
    )
    records = []
    with (
        multiprocessing.get_context("spawn").Pool(
            args.workers, initializer=worker_init, initargs=(str(args.tokenizer),)
        ) as pool,
        (args.output / "attempts.jsonl").open("x") as stream,
    ):
        for record in pool.imap(render, tasks):
            records.append(summary_record(record))
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if len(records) % 100 == 0:
                print(
                    json.dumps({"revised_bases": len(records), "target": len(tasks)}),
                    flush=True,
                )
    summary = summarize(records, revised_policy)
    if revision_identity(args.config, args.tokenizer) != identity:
        raise ValueError("assembly inputs changed during rendering")
    if has_shortfalls(summary):
        raise ValueError(
            "revised candidate failed the independent final coverage audit"
        )
    with (args.output / "attempts.jsonl").open() as stream:
        write_exports((json.loads(line) for line in stream), args.output)
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "configuration.json", revised_policy)
    atomic_json(
        args.output / "manifest.json",
        {
            "status": "completed-unsealed-candidate",
            "acceptance": "automated-checks-passed",
            "origin": origin,
            "identity": identity,
            "menu_position_policy": POSITION_VERSION,
            "selected_base_ids": sorted(selected),
            "training_authorized": False,
            "sealed": False,
            "manual_audit": "pending",
            "output_sha256": {
                path.name: digest(path)
                for path in args.output.iterdir()
                if path.is_file() and path.name != "manifest.json"
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "completed-unsealed-candidate",
                "accepted_bases": summary["accepted_bases"],
                "views_per_style": summary["views_per_style"],
                "sealed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
