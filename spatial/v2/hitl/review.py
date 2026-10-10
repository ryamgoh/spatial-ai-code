"""Immutable LLM-review packets and consensus-rejection human routing."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ReviewVerdict(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


class FinalDisposition(str, Enum):
    ACCEPT = "accept"
    HOLD_OUT = "hold-out"
    HUMAN_APPEAL = "human-appeal"


REJECTION_REASONS = frozenset(
    {
        "PROMPT_UNCLEAR",
        "CONTRACT_UNCLEAR",
        "TRACE_UNREADABLE",
        "NOT_SELF_CONTAINED",
        "REDUNDANT",
        "ANSWER_LEAKAGE",
        "OTHER_UNACCEPTABLE",
    }
)
REVIEW_CRITERIA = frozenset(
    {
        "prompt_clear",
        "contract_clear",
        "trace_readable",
        "self_contained",
        "nonredundant",
        "no_answer_leakage",
        "acceptable",
    }
)


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_verdict_reasons(
    verdict: ReviewVerdict,
    reason_codes: tuple[str, ...],
) -> None:
    unknown = set(reason_codes) - REJECTION_REASONS
    if unknown:
        raise ValueError("unknown rejection reasons: " + ", ".join(sorted(unknown)))
    if len(reason_codes) != len(set(reason_codes)):
        raise ValueError("duplicate rejection reasons")
    if verdict is ReviewVerdict.REJECT and not reason_codes:
        raise ValueError("rejection requires at least one reason")
    if verdict is ReviewVerdict.APPROVE and reason_codes:
        raise ValueError("approval cannot include rejection reasons")


@dataclass(frozen=True)
class ReviewItem:
    id: str
    prompt: str
    natural_trace: str
    payload_hash: str

    @classmethod
    def create(cls, item_id: str, prompt: str, natural_trace: str) -> ReviewItem:
        if not item_id or not prompt or not natural_trace:
            raise ValueError("review item fields must be non-empty")
        payload = {"id": item_id, "prompt": prompt, "natural_trace": natural_trace}
        return cls(item_id, prompt, natural_trace, _sha256(payload))


@dataclass(frozen=True)
class ReviewPacket:
    packet_id: str
    registry_fingerprint: str
    rubric_version: str
    items: tuple[ReviewItem, ...]
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        packet_id: str,
        registry_fingerprint: str,
        rubric_version: str,
        items: tuple[ReviewItem, ...],
    ) -> ReviewPacket:
        if not packet_id or not registry_fingerprint or not rubric_version:
            raise ValueError("review packet metadata must be non-empty")
        if not items or len({item.id for item in items}) != len(items):
            raise ValueError("review packet needs unique non-empty items")
        fingerprint = _sha256(
            {
                "packet_id": packet_id,
                "registry_fingerprint": registry_fingerprint,
                "rubric_version": rubric_version,
                "items": [asdict(item) for item in items],
            }
        )
        return cls(packet_id, registry_fingerprint, rubric_version, items, fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JudgeDecision:
    review_item_id: str
    payload_hash: str
    verdict: ReviewVerdict
    reason_codes: tuple[str, ...] = ()
    criteria: dict[str, bool] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdict", ReviewVerdict(self.verdict))
        _validate_verdict_reasons(self.verdict, self.reason_codes)
        if not isinstance(self.criteria, dict) or set(self.criteria) != REVIEW_CRITERIA:
            raise ValueError("judge decision has invalid criteria fields")
        if any(type(value) is not bool for value in self.criteria.values()):
            raise ValueError("judge criteria values must be Boolean")
        if self.verdict is ReviewVerdict.APPROVE and not all(self.criteria.values()):
            raise ValueError("approve requires every criterion to pass")
        if self.verdict is ReviewVerdict.REJECT and self.criteria["acceptable"]:
            raise ValueError("reject requires acceptable=false")


@dataclass(frozen=True)
class JudgeRun:
    provider: str
    model: str
    run_id: str
    packet_fingerprint: str
    judge_prompt_version: str
    sequence_position: int
    isolated_context: bool
    decisions: tuple[JudgeDecision, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "spatialentail-judge-run-v1", **asdict(self)}


@dataclass(frozen=True)
class RoutedReviewItem:
    item: ReviewItem
    disposition: FinalDisposition


@dataclass(frozen=True)
class ReviewRouting:
    packet_id: str
    packet_fingerprint: str
    deepseek_run_id: str
    gpt_run_id: str
    items: tuple[RoutedReviewItem, ...]


@dataclass(frozen=True)
class HumanAppeal:
    packet_id: str
    packet_fingerprint: str
    review_item_id: str
    payload_hash: str
    reviewer: str
    verdict: ReviewVerdict
    reason_codes: tuple[str, ...]
    note: str | None
    submitted_at: str

    @classmethod
    def create(
        cls,
        packet: ReviewPacket,
        review_item_id: str,
        verdict: ReviewVerdict | str,
        *,
        reviewer: str,
        reason_codes: tuple[str, ...] = (),
        note: str | None = None,
    ) -> HumanAppeal:
        verdict = ReviewVerdict(verdict)
        _validate_verdict_reasons(verdict, reason_codes)
        if not reviewer:
            raise ValueError("human appeal requires a reviewer")
        if note is not None and len(note) > 240:
            raise ValueError("human appeal note must not exceed 240 characters")
        item = next((item for item in packet.items if item.id == review_item_id), None)
        if item is None:
            raise ValueError("human appeal references an unknown review item")
        return cls(
            packet.packet_id,
            packet.fingerprint,
            item.id,
            item.payload_hash,
            reviewer,
            verdict,
            reason_codes,
            note,
            datetime.now(UTC).isoformat(),
        )


def _message(row: Mapping[str, Any], role: str) -> str:
    matches = [
        message.get("content")
        for message in row.get("messages", ())
        if message.get("role") == role
    ]
    if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0]:
        raise ValueError(f"review row needs exactly one non-empty {role} message")
    return matches[0]


def _natural_trace(row: Mapping[str, Any]) -> str:
    content = _message(row, "assistant")
    prefix = "<think>\n"
    suffix = "\n</think>\n"
    if not content.startswith(prefix) or content.count(suffix) != 1:
        raise ValueError("review row has an invalid checked Natural completion")
    trace, answer = content[len(prefix) :].split(suffix)
    if not trace or not answer.startswith("Answer:"):
        raise ValueError("review row has an invalid checked Natural completion")
    return trace


def build_review_packet(
    rows: Sequence[Mapping[str, Any]],
    *,
    packet_id: str,
    registry_fingerprint: str,
    rubric_version: str,
    sample_fraction: float = 0.02,
    minimum_per_cell: int = 5,
    seed: int = 42,
) -> tuple[ReviewPacket, dict[str, dict[str, str]]]:
    """Sample validated Natural rows and return a blinded packet plus crosswalk."""
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")
    if minimum_per_cell <= 0:
        raise ValueError("minimum_per_cell must be positive")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    source_ids = set()
    for row in rows:
        metadata = row.get("metadata", {})
        if metadata.get("round_trip_verified") is not True:
            raise ValueError("every review row must be round-trip verified")
        if metadata.get("trace_format") != "natural":
            raise ValueError("every review row must use the Natural trace format")
        if (
            metadata.get("supervision_arm") != "checked-trace"
            or metadata.get("expected_process_valid") is not True
        ):
            raise ValueError("every review row must be a valid checked trace")
        source_id = str(row.get("id", ""))
        cell = str(metadata.get("benchmark_cell", ""))
        base_id = str(metadata.get("base_id", ""))
        if not source_id or not cell or not base_id:
            raise ValueError("review rows need source, base, and benchmark-cell IDs")
        if source_id in source_ids:
            raise ValueError("review rows need unique source IDs")
        source_ids.add(source_id)
        grouped[cell].append(row)
    if not grouped:
        raise ValueError("review rows must not be empty")
    if any(len(cell_rows) < minimum_per_cell for cell_rows in grouped.values()):
        raise ValueError("every review cell must satisfy minimum_per_cell")

    rng = random.Random(seed)
    selected: list[Mapping[str, Any]] = []
    selected_ids = set()
    for cell in sorted(grouped):
        candidates = sorted(grouped[cell], key=lambda row: str(row["id"]))
        chosen = rng.sample(candidates, minimum_per_cell)
        selected.extend(chosen)
        selected_ids.update(str(row["id"]) for row in chosen)
    target = max(math.ceil(sample_fraction * len(rows)), minimum_per_cell * len(grouped))
    remaining = sorted(
        (row for row in rows if str(row["id"]) not in selected_ids),
        key=lambda row: str(row["id"]),
    )
    selected.extend(rng.sample(remaining, target - len(selected)))
    rng.shuffle(selected)

    items = []
    crosswalk = {}
    for index, row in enumerate(selected, start=1):
        item = ReviewItem.create(
            f"R{index:05d}",
            _message(row, "user"),
            _natural_trace(row),
        )
        metadata = row["metadata"]
        crosswalk[item.id] = {
            "source_id": str(row["id"]),
            "base_id": str(metadata["base_id"]),
            "benchmark_cell": str(metadata["benchmark_cell"]),
            "payload_hash": item.payload_hash,
        }
        items.append(item)
    return (
        ReviewPacket.create(
            packet_id=packet_id,
            registry_fingerprint=registry_fingerprint,
            rubric_version=rubric_version,
            items=tuple(items),
        ),
        crosswalk,
    )


def write_review_packet(
    packet: ReviewPacket,
    crosswalk: Mapping[str, Mapping[str, str]],
    packet_path: str | Path,
    crosswalk_path: str | Path,
) -> tuple[Path, Path]:
    """Persist the blinded payload separately from its audit-only crosswalk."""
    visible_path = Path(packet_path)
    sealed_path = Path(crosswalk_path)
    existing = [path for path in (visible_path, sealed_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "review output already exists: " + ", ".join(str(path) for path in existing)
        )
    expected = {item.id: item.payload_hash for item in packet.items}
    if set(crosswalk) != set(expected) or any(
        value.get("payload_hash") != expected[item_id]
        for item_id, value in crosswalk.items()
    ):
        raise ValueError("review crosswalk does not match the blinded packet")
    visible = {
        "schema": "spatialentail-quality-packet-v1",
        **packet.to_dict(),
    }
    sealed = {
        "schema": "spatialentail-quality-crosswalk-v1",
        "packet_id": packet.packet_id,
        "packet_fingerprint": packet.fingerprint,
        "items": dict(crosswalk),
    }
    visible_path.parent.mkdir(parents=True, exist_ok=True)
    visible_path.write_text(
        json.dumps(visible, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sealed_path.write_text(
        json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return visible_path, sealed_path


def load_review_packet(path: str | Path) -> ReviewPacket:
    """Load a blinded packet and verify every payload and packet fingerprint."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_keys = {
        "schema",
        "packet_id",
        "registry_fingerprint",
        "rubric_version",
        "items",
        "fingerprint",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise ValueError("review packet has invalid fields")
    if raw["schema"] != "spatialentail-quality-packet-v1":
        raise ValueError("unsupported review packet schema")
    if not isinstance(raw["items"], list):
        raise ValueError("review packet items must be a list")
    items = []
    for value in raw["items"]:
        if not isinstance(value, dict) or set(value) != {
            "id",
            "prompt",
            "natural_trace",
            "payload_hash",
        }:
            raise ValueError("review item has invalid fields")
        item = ReviewItem.create(value["id"], value["prompt"], value["natural_trace"])
        if item.payload_hash != value["payload_hash"]:
            raise ValueError(f"review item {item.id} payload hash mismatch")
        items.append(item)
    packet = ReviewPacket.create(
        packet_id=raw["packet_id"],
        registry_fingerprint=raw["registry_fingerprint"],
        rubric_version=raw["rubric_version"],
        items=tuple(items),
    )
    if packet.fingerprint != raw["fingerprint"]:
        raise ValueError("review packet fingerprint mismatch")
    return packet


def load_judge_run(
    path: str | Path,
    packet: ReviewPacket,
    *,
    expected_model: str,
) -> JudgeRun:
    """Load one model-judge receipt and verify it against the frozen packet."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_keys = {
        "schema",
        "provider",
        "model",
        "run_id",
        "packet_fingerprint",
        "judge_prompt_version",
        "sequence_position",
        "isolated_context",
        "decisions",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise ValueError("judge run has invalid fields")
    if raw["schema"] != "spatialentail-judge-run-v1":
        raise ValueError("unsupported judge-run schema")
    if raw["model"] != expected_model:
        raise ValueError(
            f"judge run expected model {expected_model}, got {raw['model']}"
        )
    if not isinstance(raw["decisions"], list):
        raise ValueError("judge decisions must be a list")
    decisions = []
    for value in raw["decisions"]:
        if not isinstance(value, dict) or set(value) != {
            "review_item_id",
            "payload_hash",
            "verdict",
            "reason_codes",
            "criteria",
        }:
            raise ValueError("judge decision has invalid fields")
        reasons = value["reason_codes"]
        if not isinstance(reasons, list):
            raise ValueError("judge decision reasons must be a list")
        decisions.append(
            JudgeDecision(
                value["review_item_id"],
                value["payload_hash"],
                ReviewVerdict(value["verdict"]),
                tuple(reasons),
                value["criteria"],
            )
        )
    run = JudgeRun(
        provider=raw["provider"],
        model=raw["model"],
        run_id=raw["run_id"],
        packet_fingerprint=raw["packet_fingerprint"],
        judge_prompt_version=raw["judge_prompt_version"],
        sequence_position=raw["sequence_position"],
        isolated_context=raw["isolated_context"],
        decisions=tuple(decisions),
    )
    _decisions_by_item(packet, run)
    return run


def _decisions_by_item(packet: ReviewPacket, run: JudgeRun) -> dict[str, JudgeDecision]:
    if not run.isolated_context:
        raise ValueError(f"judge {run.model} did not use an isolated context")
    if run.packet_fingerprint != packet.fingerprint:
        raise ValueError(f"judge {run.model} used a different review packet")
    expected = {item.id: item for item in packet.items}
    decisions = {decision.review_item_id: decision for decision in run.decisions}
    if len(decisions) != len(run.decisions) or set(decisions) != set(expected):
        raise ValueError(f"judge {run.model} must provide exactly one decision per item")
    for item_id, decision in decisions.items():
        if decision.payload_hash != expected[item_id].payload_hash:
            raise ValueError(f"judge {run.model} payload hash mismatch for {item_id}")
    return decisions


def route_review_decisions(
    packet: ReviewPacket,
    deepseek: JudgeRun,
    gpt: JudgeRun,
) -> ReviewRouting:
    """Route two independent judge runs without exposing their reasons to humans."""
    if (
        deepseek.provider != "trae"
        or gpt.provider != "trae"
        or deepseek.model != "DeepSeek-V4-Pro"
        or gpt.model != "GPT-5.5"
        or deepseek.sequence_position != 1
        or gpt.sequence_position != 2
    ):
        raise ValueError("judge order must be DeepSeek-V4-Pro then GPT-5.5")
    if deepseek.judge_prompt_version != gpt.judge_prompt_version:
        raise ValueError("judge runs must use the same judge prompt version")
    if not deepseek.run_id or not gpt.run_id or deepseek.run_id == gpt.run_id:
        raise ValueError("judge runs need distinct non-empty run IDs")
    first = _decisions_by_item(packet, deepseek)
    second = _decisions_by_item(packet, gpt)
    routed = []
    for item in packet.items:
        verdicts = (first[item.id].verdict, second[item.id].verdict)
        if verdicts == (ReviewVerdict.APPROVE, ReviewVerdict.APPROVE):
            disposition = FinalDisposition.ACCEPT
        elif verdicts == (ReviewVerdict.REJECT, ReviewVerdict.REJECT):
            disposition = FinalDisposition.HUMAN_APPEAL
        else:
            disposition = FinalDisposition.HOLD_OUT
        routed.append(RoutedReviewItem(item, disposition))
    return ReviewRouting(
        packet.packet_id,
        packet.fingerprint,
        deepseek.run_id,
        gpt.run_id,
        tuple(routed),
    )


def pending_human_items(
    routing: ReviewRouting,
    appeals: tuple[HumanAppeal, ...],
) -> tuple[ReviewItem, ...]:
    decided = set()
    for appeal in appeals:
        if (
            appeal.packet_id != routing.packet_id
            or appeal.packet_fingerprint != routing.packet_fingerprint
        ):
            raise ValueError("human appeal belongs to a different review packet")
        if appeal.review_item_id in decided:
            raise ValueError("duplicate human appeal decision")
        decided.add(appeal.review_item_id)
    return tuple(
        routed.item
        for routed in routing.items
        if routed.disposition is FinalDisposition.HUMAN_APPEAL
        and routed.item.id not in decided
    )


def append_human_appeal(
    path: str | Path,
    routing: ReviewRouting,
    appeal: HumanAppeal,
) -> None:
    pending = {item.id for item in pending_human_items(routing, ())}
    if appeal.review_item_id not in pending:
        raise ValueError("item is not pending human appeal")
    output = Path(path)
    existing = []
    if output.exists():
        existing = [
            json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(item.get("review_item_id") == appeal.review_item_id for item in existing):
            raise ValueError("human appeal already recorded for item")
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(appeal), sort_keys=True) + "\n")


def write_review_routing(
    routing: ReviewRouting,
    receipt_path: str | Path,
    human_queue_path: str | Path,
) -> tuple[Path, Path]:
    """Persist aggregate routing and a model-reason-free human appeal queue."""
    receipt = Path(receipt_path)
    queue = Path(human_queue_path)
    existing = [path for path in (receipt, queue) if path.exists()]
    if existing:
        raise FileExistsError(
            "review routing output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    counts = {disposition.value: 0 for disposition in FinalDisposition}
    for item in routing.items:
        counts[item.disposition.value] += 1
    receipt_payload = {
        "schema": "spatialentail-quality-routing-v1",
        "packet_id": routing.packet_id,
        "packet_fingerprint": routing.packet_fingerprint,
        "deepseek_run_id": routing.deepseek_run_id,
        "gpt_run_id": routing.gpt_run_id,
        "counts": counts,
        "items": [
            {
                "review_item_id": item.item.id,
                "payload_hash": item.item.payload_hash,
                "disposition": item.disposition.value,
            }
            for item in routing.items
        ],
    }
    queue_payload = {
        "schema": "spatialentail-human-appeal-queue-v1",
        "packet_id": routing.packet_id,
        "packet_fingerprint": routing.packet_fingerprint,
        "items": [
            asdict(item.item)
            for item in routing.items
            if item.disposition is FinalDisposition.HUMAN_APPEAL
        ],
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(receipt_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    queue.write_text(
        json.dumps(queue_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt, queue
