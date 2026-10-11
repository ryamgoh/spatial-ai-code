"""Rebind prior judgments only when the visible review items are identical."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

from spatial.v2.hitl.review import load_judge_run, load_review_packet
from spatial.v2.study.review_packets import prepare_review


def reuse_reviews(release: Path, reviewed: Path, output: Path) -> dict:
    report = prepare_review(release, output)
    index_path = Path(report["index"])
    index = json.loads(index_path.read_text())
    reused = 0
    provenance = output / "provenance"
    provenance.mkdir()
    for batch in index["batches"]:
        packet_path = output / batch["packet"]
        packet = load_review_packet(packet_path)
        source_packet = reviewed / batch["packet"]
        if not source_packet.is_file():
            raise FileNotFoundError(f"missing reviewed packet: {batch['packet']}")
        prior = load_review_packet(source_packet)
        if (
            prior.packet_id != packet.packet_id
            or prior.rubric_version != packet.rubric_version
            or prior.items != packet.items
        ):
            raise ValueError(
                f"reviewed items differ from final release: {batch['packet']}"
            )
        source_names = {"packet": "source-" + batch["packet"]}
        shutil.copyfile(source_packet, provenance / source_names["packet"])
        for key, model in (("deepseek", "DeepSeek-V4-Pro"), ("gpt", "GPT-5.5")):
            source = reviewed / batch[key]
            if not source.is_file():
                raise FileNotFoundError(f"missing judge receipt: {batch[key]}")
            prior_run = load_judge_run(source, prior, expected_model=model)
            rebound = replace(prior_run, packet_fingerprint=packet.fingerprint)
            (output / batch[key]).write_text(
                json.dumps(rebound.to_dict(), indent=2, sort_keys=True) + "\n"
            )
            load_judge_run(output / batch[key], packet, expected_model=model)
            source_names[key] = "source-" + batch[key]
            shutil.copyfile(source, provenance / source_names[key])
        batch["receipt_reuse"] = {
            "schema": "spatialentail-receipt-reuse-v1",
            "basis": "identical-review-items",
            "source_packet": "provenance/" + source_names["packet"],
            "source_deepseek": "provenance/" + source_names["deepseek"],
            "source_gpt": "provenance/" + source_names["gpt"],
        }
        reused += len(packet.items)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return {**report, "validated_reused_items": reused, "source": str(reviewed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reuse_reviews(args.release, args.reviewed, args.out), indent=2))


if __name__ == "__main__":
    main()
