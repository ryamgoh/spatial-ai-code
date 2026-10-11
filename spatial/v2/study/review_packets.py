"""Freeze role-stratified, bounded judge packets for one candidate release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial.v2.hitl.batches import batch_review_packet
from spatial.v2.hitl.review import build_review_packet
from spatial.v2.study.seal import audit_release


def prepare_review(release: Path, output: Path, *, seed: int = 42) -> dict:
    if output.exists():
        raise FileExistsError(f"review output exists: {output}")
    audit = audit_release(release)
    manifest = json.loads((release / "manifest.json").read_text())
    batches = []
    serialized = {}
    sample_total = 0
    for role in manifest["quotas"]:
        if role == "bench-depth":
            continue
        rows = [
            json.loads(line)
            for line in (release / (role + ".jsonl")).read_text().splitlines()
        ]
        natural = [
            r
            for r in rows
            if r["metadata"]["trace_format"] == "natural"
            and r["metadata"]["supervision_arm"] == "checked-trace"
        ]
        packet, crosswalk = build_review_packet(
            natural,
            packet_id=f"{manifest['name']}-{role}-quality-v1",
            registry_fingerprint=manifest["config_sha256"],
            rubric_version="quality-judge-v1",
            sample_fraction=0.02,
            minimum_per_cell=5,
            seed=seed,
        )
        sample_total += len(packet.items)
        for batch in batch_review_packet(packet, max_items=10, max_characters=60000):
            stem = batch.packet_id
            serialized[stem + ".json"] = {
                "schema": "spatialentail-quality-packet-v1",
                **batch.to_dict(),
            }
            serialized[stem + "-crosswalk.json"] = {
                "schema": "spatialentail-quality-crosswalk-v1",
                "packet_id": batch.packet_id,
                "packet_fingerprint": batch.fingerprint,
                "items": {item.id: crosswalk[item.id] for item in batch.items},
            }
            batches.append(
                {
                    "packet": stem + ".json",
                    "crosswalk": stem + "-crosswalk.json",
                    "deepseek": stem + "-deepseek.json",
                    "gpt": stem + "-gpt.json",
                }
            )
    index = {
        "schema": "spatialentail-release-review-v1",
        "release_artifacts_sha256": audit["release_artifacts_sha256"],
        "minimum_review_items": sample_total,
        "sampling_seed": seed,
        "sampling": "role-stratified max(2%,5 per cell)",
        "batches": batches,
    }
    output.mkdir(parents=True)
    for name, data in serialized.items():
        (output / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    return {
        "sample_items": sample_total,
        "batches": len(batches),
        "index": str(output / "index.json"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("release", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare_review(args.release, args.out), indent=2))


if __name__ == "__main__":
    main()
