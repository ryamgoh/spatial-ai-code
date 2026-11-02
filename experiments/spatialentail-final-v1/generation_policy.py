"""Executable final-v1 quotas; no data, model loading or filesystem side effects."""

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

VERSION = "matched-controlled-uniform-levels-v2"
KINDS = ("direction", "which", "count")
SPLITS = {"train": 6000, "dev": 1200, "in-range-test": 2400, "extrapolation-test": 1200}
BUILD_ORDER = ("extrapolation-test", "in-range-test", "dev", "train")
MODEL = {
    "model_id": "Qwen/Qwen3-4B-Thinking-2507",
    "revision": "768f209d9ea81521153ed38c47d515654e938aea",
    "context_length": 8192,
    "stage1_max_new_tokens": 4096,
    "stage2_max_new_tokens": 16,
}
ALTERNATIVES = 5
SEED = 42
SIZE_TABLE = {
    str(level): {
        "objects": [2 * level + 4, 2 * level + 8],
        "premises": [2 * level + 6, 2 * level + 14],
    }
    for level in range(1, 11)
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stable_int(text):
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def levels(split):
    return (9, 10) if split == "extrapolation-test" else tuple(range(1, 9))


def apportioned(total, count):
    return [total // count + (i < total % count) for i in range(count)]


def case_counts(kind, total):
    counts = {"A-D": total * 60 // 100, "F": total // 10}
    if kind == "which":
        counts.update(
            multiple=total * 15 // 100,
            ambiguous=total * 45 // 400,
            inconsistent=total * 15 // 400,
        )
    else:
        counts.update(ambiguous=total * 45 // 200, inconsistent=total * 15 // 200)
    require(sum(counts.values()) == total, "nonintegral split proportions")
    return counts


def level_cases(kind, total, count):
    """Exact uniform level margins; E remainders go to the least-filled level."""
    targets = case_counts(kind, total)
    rows = [
        {"A-D": targets["A-D"] // count, "F": targets["F"] // count}
        for _ in range(count)
    ]
    e_capacity = [total * 30 // 100 // count] * count
    e_cases = [c for c in targets if c not in ("A-D", "F")]
    for case in e_cases:
        base = targets[case] // count
        for i, row in enumerate(rows):
            row[case] = base
            e_capacity[i] -= base
    for case in e_cases:
        for i in sorted(range(count), key=lambda i: (-e_capacity[i], i))[
            : targets[case] % count
        ]:
            rows[i][case] += 1
            e_capacity[i] -= 1
    require(not any(e_capacity), "E allocation cannot satisfy level margins")
    require(
        all(sum(r.values()) == total // count for r in rows),
        "nonuniform level allocation",
    )
    return rows


@dataclass(frozen=True)
class Slot:
    split: str
    kind: str
    level: int
    case: str
    index: int
    cohort: str
    count_value: int = 1
    contingent: bool = False
    answer_position: str | None = None

    def identity(self, alternative):
        return (
            f"final-v1-{self.split}-{self.kind}-l{self.level:02d}-"
            f"{self.case}-{self.index:04d}-a{alternative}"
        )

    def seed(self, alternative):
        return stable_int(f"{SEED}/{self.identity(alternative)}")

    @property
    def requested_depth(self):
        return self.level if self.cohort == "depth" else None

    @property
    def scene_case(self):
        return {
            "A-D": "sole",
            "F": "sole",
            "ambiguous": "no-entailed",
            "multiple": "multiple",
            "inconsistent": "inconsistent",
        }[self.case]


def schedule():
    slots = []
    for split in BUILD_ORDER:
        total = SPLITS[split] // 3
        ls = levels(split)
        for kind in KINDS:
            table = level_cases(kind, total, len(ls))
            semantic_positive = apportioned(total * 15 // 100, len(ls))
            # 2% of Which questions: at least one contingent case at every level.
            contingent = apportioned(total * 2 // 100, len(ls))
            position = 0
            for li, level in enumerate(ls):
                for case, number in table[li].items():
                    for i in range(number):
                        semantic = case in ("ambiguous", "inconsistent") or (
                            kind == "which"
                            and case == "A-D"
                            and i < semantic_positive[li]
                        )
                        is_contingent = (
                            kind == "which" and case == "A-D" and i < contingent[li]
                        )
                        letter = "ABCD"[position % 4] if case == "A-D" else None
                        if letter is not None:
                            position += 1
                        slots.append(
                            Slot(
                                split,
                                kind,
                                level,
                                case,
                                i,
                                "semantic" if semantic else "depth",
                                i % 5 if kind == "count" else 1,
                                is_contingent,
                                letter,
                            )
                        )
    validate_schedule(slots)
    return slots


def validate_schedule(slots):
    require(len({s.identity(0) for s in slots}) == len(slots), "duplicate slot IDs")
    for split, size in SPLITS.items():
        for kind in KINDS:
            rows = [s for s in slots if (s.split, s.kind) == (split, kind)]
            n = size // 3
            require(
                Counter(s.case for s in rows) == Counter(case_counts(kind, n)),
                f"wrong case allocation: {split}/{kind}",
            )
            require(
                Counter(s.level for s in rows)
                == Counter({l: n // len(levels(split)) for l in levels(split)}),
                "wrong construction-level totals",
            )
            require(
                Counter(s.answer_position for s in rows if s.case == "A-D")
                == Counter(dict.fromkeys("ABCD", n * 60 // 100 // 4)),
                "unbalanced correct option positions",
            )
            require(
                sum(s.cohort == "depth" for s in rows) == n * 70 // 100,
                "wrong evidence-cohort allocation",
            )
            for level in levels(split):
                cell = [s for s in rows if s.level == level]
                require(
                    {s.case for s in cell} == set(case_counts(kind, n)),
                    "missing case at construction level",
                )
                if kind == "which":
                    require(any(s.contingent for s in cell), "missing contingent case")


def smoke_schedule(slots):
    chosen = {}
    for s in slots:
        if s.level in (1, 8, 9, 10):
            key = (
                s.split,
                s.kind,
                s.level,
                s.case,
                s.contingent,
                s.count_value if s.case in ("A-D", "F") else None,
            )
            chosen.setdefault(key, s)
    return list(chosen.values())


def policy_manifest():
    return {
        "version": VERSION,
        "seed": SEED,
        "split_bases": SPLITS,
        "export_splits": {
            "in-range-test": "sft-bench",
            "extrapolation-test": "sft-bench",
        },
        "external_bases": 1500,
        "external_scenes": 500,
        "menus_per_base": 1,
        "menu_policy_version": "synthetic-count-options-0-4-v1",
        "synthetic_count_menu_values": list(range(5)),
        "scene_policy_version": "explicit-size-checked-augmentation-v1",
        "size_policy": {
            "version": "linear-coupled-sizes-v1",
            # Preserve the sampling stream used by the successful feasibility run.
            "seed_namespace": "flexible-core-checked-augmentation-probe-v1/size",
            "levels": SIZE_TABLE,
            "premises_minus_objects": [2, 6],
        },
        "augmentation_candidate_limit": 512,
        "multiple_which_entailed_objects": 2,
        "training_duplicate_policy": "last-attempt-within-train-only-v1",
        "construction_candidates_per_slot": ALTERNATIVES,
        "which_contingent_fraction": "2/100",
        "model": MODEL,
        "admission_tokenizer_policy": "pinned-training-tokenizer-v1",
        "level_case_tables": {
            split: {
                kind: dict(
                    zip(
                        map(str, levels(split)),
                        level_cases(kind, n // 3, len(levels(split))),
                    )
                )
                for kind in KINDS
            }
            for split, n in SPLITS.items()
        },
        "slots": [asdict(s) for s in schedule()],
    }


def workflow_config():
    """Supply experiment policy and source locations to the shared workflow."""
    from spatial.v2.entailment_generation import GenerationConfig

    experiment = Path(__file__).resolve().parent
    root = experiment.parents[1]
    correction = root / "experiments/spatialentail-randomized-v2/prepared-corr-eval"
    return GenerationConfig(
        root=root,
        experiment=experiment,
        policy=policy_manifest(),
        slot_type=Slot,
        smoke_slots=tuple(smoke_schedule(schedule())),
        coverage_policy=root
        / (
            "experiments/spatialentail-composition-v1/"
            "CONTROLLED-TRAINING-BUILD-v2.manifest.json"
        ),
        original_source=root / "data/spatialeval_org.jsonl",
        corr_source=correction / "data/corr.jsonl",
        correction_receipt=correction / "correction-summary.json",
        pending_release_checks=(
            "nuisance-shortcut-probes-and-paired-example-review",
            "actual-axolotl-loader-collator-and-gpu-validation",
            "final-v1-training-and-epoch-evaluation-integration",
        ),
        anchor_prefix="final-anchor",
        family_prefix="final-v1",
        manifest_schema="spatialentail-final-candidate-v1",
    )
