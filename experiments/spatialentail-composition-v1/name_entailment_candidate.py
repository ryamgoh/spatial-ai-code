"""Apply ordinary names to verified bases; preserve scene and selection identities."""

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

from spatial.v2.composition_execution import atomic_json
from spatial.v2.composition_production import summary_record, write_exports
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.entailment_names import NAMES, VERSION, rename_record
from spatial.v2.entailment_production import has_shortfalls, summarize

_tokenizer = None


def initialize(path):
    global _tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)


def rename(record):
    return rename_record(record, _tokenizer)


def code_hashes():
    return {path.name: digest(path) for path in sorted(Path("spatial/v2").glob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh output directory")
    if args.tokenizer.name != REVISION or not 1 <= args.workers <= 128:
        parser.error("provide the pinned tokenizer and 1-128 workers")
    source_manifest = args.source / "manifest.json"
    original = json.loads(source_manifest.read_text())
    if original["status"] != "completed-unsealed-candidate" or original["sealed"]:
        raise ValueError("naming requires a completed unsealed source")
    for name, sha in original["output_sha256"].items():
        if digest(args.source / name) != sha:
            raise ValueError("source artifact integrity changed")
    policy = json.loads((args.source / "configuration.json").read_text())
    snapshot = code_hashes()
    manifest = {
        "status": "naming-unsealed-candidate",
        "sealed": False,
        "training_authorized": False,
        "manual_audit": "pending-named-reference-review",
        "source": str(args.source),
        "source_manifest_sha256": digest(source_manifest),
        "naming_policy": VERSION,
        "vocabulary": list(NAMES),
        "sources_sha256": snapshot,
        "tokenizer_sha256": {
            p.name: digest(p) for p in args.tokenizer.iterdir() if p.is_file()
        },
        "labels_base_ids_families_splits_changed": False,
    }
    args.output.mkdir()
    atomic_json(args.output / "manifest.json", manifest)
    started = time.perf_counter()
    records, story_maps = [], {}
    try:
        with (
            (args.source / "attempts.jsonl").open() as source,
            (args.output / "attempts.jsonl").open("x") as output,
            multiprocessing.get_context("spawn").Pool(
                args.workers, initializer=initialize, initargs=(str(args.tokenizer),)
            ) as pool,
        ):
            tasks = (json.loads(line) for line in source)
            for record in pool.imap(rename, tasks):
                old_map = story_maps.setdefault(
                    record["story_id"], record["entity_name_map"]
                )
                if old_map != record["entity_name_map"]:
                    raise ValueError("query types do not share one story naming map")
                output.write(json.dumps(record) + "\n")
                output.flush()
                records.append(summary_record(record))
                if len(records) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "named_bases": len(records),
                                "elapsed_seconds": round(
                                    time.perf_counter() - started, 2
                                ),
                            }
                        ),
                        flush=True,
                    )
        if code_hashes() != snapshot:
            raise ValueError("naming software changed during the revision")
        policy["naming_policy"] = {
            "version": VERSION,
            "vocabulary": list(NAMES),
            "mapping_unit": "story",
            "source_manifest_sha256": digest(source_manifest),
        }
        policy["frozen_sources_sha256"] = snapshot
        summary = summarize(records, policy)
        if has_shortfalls(summary):
            raise ValueError("naming changed coverage or quota acceptance")
        with (args.output / "attempts.jsonl").open() as source:
            write_exports((json.loads(line) for line in source), args.output)
        atomic_json(args.output / "configuration.json", policy)
        atomic_json(args.output / "summary.json", summary)
        manifest.update(
            status="completed-unsealed-candidate",
            acceptance="automated-checks-passed",
            named_bases=len(records),
            named_stories=len(story_maps),
            elapsed_seconds=round(time.perf_counter() - started, 2),
            output_sha256={
                p.name: digest(p)
                for p in args.output.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        )
        atomic_json(args.output / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "named_bases": len(records),
                    "views_per_style": summary["views_per_style"],
                    "sealed": False,
                }
            )
        )
    except BaseException as error:
        manifest.update(
            status="failed-unsealed-naming-revision",
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
