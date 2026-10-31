"""Generate audited training questions and prepare both arms from tracked inputs."""

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import build_controlled_training as build
import run_all_controlled_prototype as tokenizer_loader
from rerender_controlled_training import rerender

from spatial.v2.entailment_replacement_allocation import (
    policy_from,
    schedule,
    validate_records,
)
from spatial.v2.entailment_thinking_training import (
    EXPERIMENT,
    ROOT,
    digest,
    prepare,
    require,
    write_json,
)

EXCLUSION_SHA = "c19250414a4f868b3519927fe237dc11c2b28f26d414eaa7472fa824def6c281"


def inputs():
    exclusion_path = EXPERIMENT / "generation-exclusions-v1.json"
    require(digest(exclusion_path) == EXCLUSION_SHA, "evaluation exclusions changed")
    held_out = json.loads(exclusion_path.read_text())
    selection = json.loads(
        (EXPERIMENT / "unique-only-selection-v1/selection.json").read_text()
    )
    require(
        len(held_out["base_ids"]) == len(set(held_out["base_ids"])) == 4800
        and set(held_out["base_ids"]) <= set(selection["selected_base_ids"]),
        "fixed evaluation inventory mismatch",
    )
    frozen = json.loads((EXPERIMENT / "entailment-config-frozen-v1.json").read_text())
    policy = policy_from(frozen)
    original = json.loads(
        (EXPERIMENT / "CONTROLLED-TRAINING-BUILD-v2.manifest.json").read_text()
    )
    require(policy == original["policy"], "approved allocation changed")
    return policy, held_out


def generate_one(job):
    admitted = build.admit(job)
    require(
        admitted["status"] == "accepted",
        f"generation admission failed: {admitted.get('reason', admitted['status'])}",
    )
    result = rerender(admitted["record"])
    require(result["status"] == "passed", f"prose audit failed: {result.get('reason')}")
    return result["record"]


def generate(output, workers):
    require(not output.exists(), "refusing to overwrite generation output")
    require(1 <= workers <= 4, "workers must be between one and four")
    require(output.parent.is_dir(), "output parent must exist")
    policy, held_out = inputs()
    source_paths = [
        *sorted((ROOT / "spatial/v2").glob("*.py")),
        *sorted(EXPERIMENT.glob("*.py")),
        EXPERIMENT / "generation-exclusions-v1.json",
        EXPERIMENT / "entailment-config-frozen-v1.json",
        EXPERIMENT / "CONTROLLED-TRAINING-BUILD-v2.manifest.json",
        EXPERIMENT / "unique-only-selection-v1/selection.json",
    ]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in source_paths}
    jobs, preflight = build.preflight(schedule(policy), held_out)
    require(not preflight["shortages"] and len(jobs) == 6000, "fixed supply shortage")
    receipt = output.parent / "generation.json"
    require(not receipt.exists(), "existing generation receipt; use a fresh run")
    write_json(
        receipt,
        {
            "schema": "portable-controlled-generation-v1",
            "status": "running",
            "inputs_sha256": hashes,
            "preflight": preflight,
            "construction_cap": 18000,
            "solver_admission_cap": 6000,
            "refill": False,
            "policy": policy,
        },
    )

    # Retain the audited generator's historical admission-token gate, then
    # independently tokenize the final training data with the active model.
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        "Qwen/Qwen3.5-4B",
        revision=tokenizer_loader.REVISION,
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
    records = []
    with (
        get_context("spawn").Pool(
            workers, initializer=tokenizer_loader.initialize, initargs=(snapshot,)
        ) as pool,
        (output.parent / "generation-records.jsonl").open("x") as journal,
    ):
        for record in pool.imap(generate_one, jobs):
            journal.write(json.dumps(record) + "\n")
            journal.flush()
            records.append(record)
            if len(records) % 100 == 0:
                print(f"Generated and audited {len(records)}/6000", flush=True)
    validate_records(records, policy)
    require(
        not ({r["query_key"] for r in records} & set(held_out["query_keys"])),
        "evaluation query overlap",
    )
    require(
        not ({r["scene_key"] for r in records} & set(held_out["scene_keys"])),
        "evaluation scene overlap",
    )
    require(
        not ({r["anchor_key"] for r in records} & set(held_out["anchor_keys"])),
        "evaluation family overlap",
    )
    require(all(digest(ROOT / p) == sha for p, sha in hashes.items()), "inputs changed")
    generation = json.loads(receipt.read_text())
    generation.update(status="passed", records=6000)
    write_json(receipt, generation)
    build.export(
        records,
        output,
        {
            "status": "references-rerendered-and-audited",
            "statuses": {"passed": 6000},
            "failures": [],
            "question_generation": True,
            "generation_schema": "portable-controlled-generation-v1",
        },
        receipt,
    )
    write_json(output / "generation.json", generation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    require(not args.prepared.exists(), "prepared output already exists")
    candidate = args.prepared.parent / "candidate"
    generate(candidate, args.workers)
    prepare(EXPERIMENT / "active-model.json", args.prepared, candidate=candidate)


if __name__ == "__main__":
    main()
