"""Immutable bounded judge packets and validated aggregate dispositions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from spatial.v2.hitl.review import ReviewPacket, ReviewRouting


def batch_review_packet(
    packet: ReviewPacket, *, max_items: int = 10, max_characters: int = 60000
) -> tuple[ReviewPacket, ...]:
    if max_items < 1 or max_characters < 1:
        raise ValueError("batch limits must be positive")
    chunks = []
    current = []
    characters = 0
    for item in packet.items:
        size = len(item.prompt) + len(item.natural_trace)
        if size > max_characters:
            raise ValueError(f"{item.id} exceeds single-item review budget")
        if current and (
            len(current) == max_items or characters + size > max_characters
        ):
            chunks.append(tuple(current))
            current, characters = [], 0
        current.append(item)
        characters += size
    if current:
        chunks.append(tuple(current))
    return tuple(
        ReviewPacket.create(
            packet_id=f"{packet.packet_id}-batch-{index:03d}",
            registry_fingerprint=packet.registry_fingerprint,
            rubric_version=packet.rubric_version,
            items=chunk,
        )
        for index, chunk in enumerate(chunks, 1)
    )


def aggregate_review_batches(
    parent: ReviewPacket,
    batches: Sequence[ReviewPacket],
    routings: Sequence[ReviewRouting],
) -> dict:
    if len(batches) != len(routings) or not batches:
        raise ValueError("exactly one routing is required per review batch")
    expected = {item.id: item.payload_hash for item in parent.items}
    seen = {}
    counts = Counter()
    for batch, routing in zip(batches, routings, strict=True):
        if (
            routing.packet_id != batch.packet_id
            or routing.packet_fingerprint != batch.fingerprint
        ):
            raise ValueError("routing belongs to a different review batch")
        if (
            batch.registry_fingerprint != parent.registry_fingerprint
            or batch.rubric_version != parent.rubric_version
        ):
            raise ValueError("batch protocol disagrees with parent")
        batch_items = {item.id: item.payload_hash for item in batch.items}
        if len(routing.items) != len(batch.items):
            raise ValueError("routing is incomplete")
        for routed in routing.items:
            item = routed.item
            if item.id in seen or batch_items.get(item.id) != item.payload_hash:
                raise ValueError("duplicate or altered batch item")
            seen[item.id] = item.payload_hash
            counts[routed.disposition.value] += 1
    if seen != expected:
        raise ValueError("review batches do not cover the frozen parent packet")
    return {
        "schema": "spatialentail-batched-review-v1",
        "packet_id": parent.packet_id,
        "packet_fingerprint": parent.fingerprint,
        "counts": dict(counts),
        "batches": [
            {
                "packet_id": batch.packet_id,
                "fingerprint": batch.fingerprint,
                "deepseek_run_id": routing.deepseek_run_id,
                "gpt_run_id": routing.gpt_run_id,
            }
            for batch, routing in zip(batches, routings, strict=True)
        ],
    }
