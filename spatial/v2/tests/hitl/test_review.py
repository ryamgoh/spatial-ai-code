"""Public-contract tests for sequential quality-review routing."""

from __future__ import annotations

import json

import pytest

from spatial.v2.hitl import (
    FinalDisposition,
    HumanAppeal,
    JudgeDecision,
    JudgeRun,
    ReviewItem,
    ReviewPacket,
    ReviewVerdict,
    append_human_appeal,
    build_review_packet,
    load_judge_run,
    load_review_packet,
    write_review_packet,
    pending_human_items,
    route_review_decisions,
    write_review_routing,
)


def _natural_row(index: int, cell: str) -> dict:
    return {
        "id": f"source-{index}",
        "messages": [
            {"role": "system", "content": "Answer the question."},
            {"role": "user", "content": f"Prompt {index}"},
            {
                "role": "assistant",
                "content": f"<think>\nTrace {index}\n</think>\nAnswer: A",
            },
        ],
        "metadata": {
            "base_id": f"base-{index}",
            "benchmark_cell": cell,
            "trace_format": "natural",
            "supervision_arm": "checked-trace",
            "expected_process_valid": True,
            "round_trip_verified": True,
        },
    }


def _packet() -> ReviewPacket:
    return ReviewPacket.create(
        packet_id="quality-v1",
        registry_fingerprint="registry-sha256",
        rubric_version="readability-v1",
        items=(
            ReviewItem.create("opaque-1", "Prompt 1", "Trace 1"),
            ReviewItem.create("opaque-2", "Prompt 2", "Trace 2"),
            ReviewItem.create("opaque-3", "Prompt 3", "Trace 3"),
        ),
    )


def _run(
    packet: ReviewPacket,
    model: str,
    verdicts: tuple[ReviewVerdict, ...],
) -> JudgeRun:
    approved_criteria = {
        "prompt_clear": True,
        "contract_clear": True,
        "trace_readable": True,
        "self_contained": True,
        "nonredundant": True,
        "no_answer_leakage": True,
        "acceptable": True,
    }
    return JudgeRun(
        provider="trae",
        model=model,
        run_id=f"run-{model}",
        packet_fingerprint=packet.fingerprint,
        judge_prompt_version="quality-judge-v1",
        sequence_position=1 if model == "DeepSeek-V4-Pro" else 2,
        isolated_context=True,
        decisions=tuple(
            JudgeDecision(
                item.id,
                item.payload_hash,
                verdict,
                ("OTHER_UNACCEPTABLE",)
                if verdict is ReviewVerdict.REJECT
                else (),
                {
                    **approved_criteria,
                    **(
                        {"acceptable": False}
                        if verdict is ReviewVerdict.REJECT
                        else {}
                    ),
                },
            )
            for item, verdict in zip(packet.items, verdicts, strict=True)
        ),
    )


def test_review_routing_accepts_only_consensus_and_appeals_double_rejection() -> None:
    packet = _packet()
    deepseek = _run(
        packet,
        "DeepSeek-V4-Pro",
        (ReviewVerdict.APPROVE, ReviewVerdict.APPROVE, ReviewVerdict.REJECT),
    )
    gpt = _run(
        packet,
        "GPT-5.5",
        (ReviewVerdict.APPROVE, ReviewVerdict.REJECT, ReviewVerdict.REJECT),
    )

    routing = route_review_decisions(packet, deepseek, gpt)

    assert [item.disposition for item in routing.items] == [
        FinalDisposition.ACCEPT,
        FinalDisposition.HOLD_OUT,
        FinalDisposition.HUMAN_APPEAL,
    ]
    assert [item.id for item in pending_human_items(routing, ())] == ["opaque-3"]


def test_review_routing_rejects_nonisolated_or_incomplete_judge_runs() -> None:
    packet = _packet()
    complete = _run(packet, "GPT-5.5", (ReviewVerdict.APPROVE,) * 3)
    nonisolated = JudgeRun(
        provider="trae",
        model="DeepSeek-V4-Pro",
        run_id="run-deepseek",
        packet_fingerprint=packet.fingerprint,
        judge_prompt_version="quality-judge-v1",
        sequence_position=1,
        isolated_context=False,
        decisions=complete.decisions,
    )
    with pytest.raises(ValueError, match="isolated context"):
        route_review_decisions(packet, nonisolated, complete)

    incomplete = JudgeRun(
        provider="trae",
        model="DeepSeek-V4-Pro",
        run_id="run-deepseek",
        packet_fingerprint=packet.fingerprint,
        judge_prompt_version="quality-judge-v1",
        sequence_position=1,
        isolated_context=True,
        decisions=complete.decisions[:-1],
    )
    with pytest.raises(ValueError, match="exactly one decision"):
        route_review_decisions(packet, incomplete, complete)


def test_only_double_rejection_can_receive_a_human_appeal(tmp_path) -> None:
    packet = _packet()
    routing = route_review_decisions(
        packet,
        _run(
            packet,
            "DeepSeek-V4-Pro",
            (ReviewVerdict.APPROVE, ReviewVerdict.APPROVE, ReviewVerdict.REJECT),
        ),
        _run(
            packet,
            "GPT-5.5",
            (ReviewVerdict.APPROVE, ReviewVerdict.REJECT, ReviewVerdict.REJECT),
        ),
    )
    events = tmp_path / "human-appeals.jsonl"

    with pytest.raises(ValueError, match="not pending human appeal"):
        append_human_appeal(
            events,
            routing,
            HumanAppeal.create(
                packet,
                "opaque-2",
                ReviewVerdict.APPROVE,
                reviewer="human-1",
            ),
        )

    appeal = HumanAppeal.create(
        packet,
        "opaque-3",
        ReviewVerdict.REJECT,
        reviewer="human-1",
        reason_codes=("TRACE_UNREADABLE",),
        note="The dependency order is difficult to follow.",
    )
    append_human_appeal(events, routing, appeal)

    stored = json.loads(events.read_text().strip())
    assert stored["review_item_id"] == "opaque-3"
    assert stored["verdict"] == "reject"
    assert [item.id for item in pending_human_items(routing, (appeal,))] == []


def test_rejection_requires_reason_and_approval_forbids_one() -> None:
    packet = _packet()
    with pytest.raises(ValueError, match="rejection requires"):
        HumanAppeal.create(
            packet,
            "opaque-3",
            ReviewVerdict.REJECT,
            reviewer="human-1",
        )
    with pytest.raises(ValueError, match="approval cannot"):
        HumanAppeal.create(
            packet,
            "opaque-3",
            ReviewVerdict.APPROVE,
            reviewer="human-1",
            reason_codes=("TRACE_UNREADABLE",),
        )


def test_judge_decision_requires_complete_consistent_criteria() -> None:
    packet = _packet()
    item = packet.items[0]
    with pytest.raises(ValueError, match="criteria fields"):
        JudgeDecision(
            item.id,
            item.payload_hash,
            ReviewVerdict.APPROVE,
            (),
            {"acceptable": True},
        )
    with pytest.raises(ValueError, match="approve requires every criterion"):
        JudgeDecision(
            item.id,
            item.payload_hash,
            ReviewVerdict.APPROVE,
            (),
            {
                "prompt_clear": True,
                "contract_clear": True,
                "trace_readable": False,
                "self_contained": True,
                "nonredundant": True,
                "no_answer_leakage": True,
                "acceptable": True,
            },
        )


def test_packet_builder_samples_each_cell_and_keeps_crosswalk_sealed() -> None:
    rows = tuple(
        _natural_row(index, "DIR-ARC-UNI" if index < 5 else "SEL-IFF-UNI")
        for index in range(10)
    )

    packet, crosswalk = build_review_packet(
        rows,
        packet_id="pilot-review-v1",
        registry_fingerprint="registry-sha256",
        rubric_version="readability-v1",
        sample_fraction=0.2,
        minimum_per_cell=2,
        seed=42,
    )

    assert len(packet.items) == 4
    assert set(crosswalk) == {item.id for item in packet.items}
    assert {
        crosswalk[item.id]["benchmark_cell"] for item in packet.items
    } == {"DIR-ARC-UNI", "SEL-IFF-UNI"}
    serialized_packet = json.dumps(packet.to_dict())
    assert "source-" not in serialized_packet
    assert "base-" not in serialized_packet
    assert all(item.prompt.startswith("Prompt ") for item in packet.items)
    assert all(item.natural_trace.startswith("Trace ") for item in packet.items)


def test_packet_builder_rejects_nonvalidated_or_nonnatural_rows() -> None:
    row = _natural_row(1, "DIR-ARC-UNI")
    row["metadata"]["round_trip_verified"] = False
    with pytest.raises(ValueError, match="round-trip verified"):
        build_review_packet(
            (row,),
            packet_id="bad",
            registry_fingerprint="registry-sha256",
            rubric_version="readability-v1",
            minimum_per_cell=1,
        )


def test_review_packet_and_crosswalk_are_persisted_separately(tmp_path) -> None:
    packet, crosswalk = build_review_packet(
        tuple(_natural_row(index, "DIR-ARC-UNI") for index in range(5)),
        packet_id="pilot-review-v1",
        registry_fingerprint="registry-sha256",
        rubric_version="readability-v1",
        minimum_per_cell=2,
        seed=42,
    )
    packet_path = tmp_path / "packet.json"
    crosswalk_path = tmp_path / "crosswalk.json"

    write_review_packet(packet, crosswalk, packet_path, crosswalk_path)

    visible = json.loads(packet_path.read_text())
    sealed = json.loads(crosswalk_path.read_text())
    assert visible["fingerprint"] == packet.fingerprint
    assert "source-" not in packet_path.read_text()
    assert sealed["packet_fingerprint"] == packet.fingerprint
    assert set(sealed["items"]) == set(crosswalk)
    assert load_review_packet(packet_path) == packet
    with pytest.raises(FileExistsError, match="already exists"):
        write_review_packet(packet, crosswalk, packet_path, crosswalk_path)


def test_judge_run_loader_pins_model_packet_and_payloads(tmp_path) -> None:
    packet = _packet()
    run = _run(
        packet,
        "DeepSeek-V4-Pro",
        (ReviewVerdict.APPROVE, ReviewVerdict.REJECT, ReviewVerdict.APPROVE),
    )
    path = tmp_path / "judge.json"
    path.write_text(json.dumps(run.to_dict()))

    assert load_judge_run(path, packet, expected_model="DeepSeek-V4-Pro") == run

    wrong = json.loads(path.read_text())
    wrong["model"] = "GPT-5.5"
    path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="expected model"):
        load_judge_run(path, packet, expected_model="DeepSeek-V4-Pro")


def test_review_routing_persists_receipt_and_blinded_human_queue(tmp_path) -> None:
    packet = _packet()
    routing = route_review_decisions(
        packet,
        _run(
            packet,
            "DeepSeek-V4-Pro",
            (ReviewVerdict.APPROVE, ReviewVerdict.APPROVE, ReviewVerdict.REJECT),
        ),
        _run(
            packet,
            "GPT-5.5",
            (ReviewVerdict.APPROVE, ReviewVerdict.REJECT, ReviewVerdict.REJECT),
        ),
    )
    receipt_path = tmp_path / "routing.json"
    queue_path = tmp_path / "human-queue.json"

    write_review_routing(routing, receipt_path, queue_path)

    receipt = json.loads(receipt_path.read_text())
    queue = json.loads(queue_path.read_text())
    assert receipt["counts"] == {"accept": 1, "hold-out": 1, "human-appeal": 1}
    assert receipt["deepseek_run_id"] == "run-DeepSeek-V4-Pro"
    assert receipt["gpt_run_id"] == "run-GPT-5.5"
    assert [item["id"] for item in queue["items"]] == ["opaque-3"]
    assert "reason_codes" not in queue_path.read_text()


def test_review_routing_enforces_provider_prompt_and_sequence() -> None:
    packet = _packet()
    deepseek = _run(packet, "DeepSeek-V4-Pro", (ReviewVerdict.APPROVE,) * 3)
    gpt = _run(packet, "GPT-5.5", (ReviewVerdict.APPROVE,) * 3)

    wrong_sequence = JudgeRun(
        provider="trae",
        model=deepseek.model,
        run_id=deepseek.run_id,
        packet_fingerprint=deepseek.packet_fingerprint,
        judge_prompt_version=deepseek.judge_prompt_version,
        sequence_position=2,
        isolated_context=True,
        decisions=deepseek.decisions,
    )
    with pytest.raises(ValueError, match="judge order"):
        route_review_decisions(packet, wrong_sequence, gpt)

    wrong_prompt = JudgeRun(
        provider="trae",
        model=gpt.model,
        run_id=gpt.run_id,
        packet_fingerprint=gpt.packet_fingerprint,
        judge_prompt_version="different-prompt",
        sequence_position=2,
        isolated_context=True,
        decisions=gpt.decisions,
    )
    with pytest.raises(ValueError, match="same judge prompt"):
        route_review_decisions(packet, deepseek, wrong_prompt)
