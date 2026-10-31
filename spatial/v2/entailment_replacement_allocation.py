"""All-controlled allocation with the approved numerical depth/answer safeguards."""

from collections import Counter
from dataclasses import dataclass

from spatial.v2 import composition_production as historical
from spatial.v2.composition_production import allocate
from spatial.v2.entailment_retention import matches
from spatial.v2.entailment_training_selection import answer_category
from spatial.v2.entailment_unique_study import training_allocation_policy

VERSION = "all-controlled-training-allocation-v1"
CONSTRUCTION_CANDIDATES_PER_SLOT = 3


def policy_from(frozen):
    old = training_allocation_policy(frozen)
    shared = {**old, "schema": historical.SCHEMA}
    return {
        "version": VERSION,
        "training_allocation": old["training_allocation"],
        "strata": {
            k: v for k, v in historical.quotas(shared).items() if k.startswith("train/")
        },
        "count_targets": {
            k: v
            for k, v in historical.count_quotas(shared).items()
            if k.startswith("train/")
        },
        "depth_family_minima": {
            k: v
            for k, v in historical.depth_family_minima(shared).items()
            if k.startswith("train/")
        },
        "coverage_minima": [
            r for r in old["coverage_minima"] if r["match"].get("split") == "train"
        ],
        "depth_weights": old["depth_weights"]["train"],
        "source": "controlled",
        "source_change": (
            "All construction is controlled. Retired 70/15/10/5 generation-source "
            "quotas are not enforced. The 1400-depth/600-semantic per-query "
            "strata are retained as evidence cohorts, not source labels. "
            "All numerical depth, Count-value, family and evidence minima are unchanged."
        ),
        "which_cohorts": (
            "Depth cohort: 900 A-D + 200 F + 300 multiple-entailment E. "
            "Semantic cohort: 300 A-D + 225 no-entailed E + 75 inconsistent E. "
            "Extra positive support in the semantic cohort receives no depth-stratum credit."
        ),
        "construction_candidates_per_slot": CONSTRUCTION_CANDIDATES_PER_SLOT,
        "training_authorized": False,
    }


@dataclass(frozen=True)
class Slot:
    index: int
    kind: str
    category: str
    span: int
    depth: int | None
    count_value: int = 1
    contingent: bool = False

    @property
    def stratum(self):
        return (
            f"train/{self.kind}/{self.depth if self.depth is not None else 'semantic'}"
        )

    @property
    def case(self):
        return {
            "A-D": "sole",
            "unique-unlisted": "sole",
            "no-entailed-answer": "no-entailed",
            "multiple-entailed": "multiple",
            "inconsistent": "inconsistent",
        }[self.category]

    def identity(self, alternative):
        return f"controlled-train-v2-q{self.index:05d}-a{alternative}"

    def seed(self, alternative):
        return 2035103100 + self.index * CONSTRUCTION_CANDIDATES_PER_SLOT + alternative


def schedule(policy):
    slots = []
    weights = policy["depth_weights"]
    omission = allocate(200, weights)
    multiple = allocate(300, weights)

    def add(kind, category, span, depth, number, count=1, contingent=False):
        for _ in range(number):
            slots.append(
                Slot(len(slots), kind, category, span, depth, count, contingent)
            )

    for kind in ("direction", "which", "count"):
        for depth in range(1, 9):
            total = policy["strata"][f"train/{kind}/{depth}"]
            f = omission[str(depth)]
            if kind == "count":
                counts = {
                    str(v): policy["count_targets"][f"train/count/{depth}/{v}"]
                    for v in range(5)
                }
                f_counts = allocate(f, counts)
                for value in range(5):
                    n_f = f_counts[str(value)]
                    add(kind, "A-D", depth, depth, counts[str(value)] - n_f, value)
                    add(kind, "unique-unlisted", depth, depth, n_f, value)
            else:
                n_multiple = multiple[str(depth)] if kind == "which" else 0
                add(kind, "A-D", depth, depth, total - f - n_multiple)
                add(kind, "unique-unlisted", depth, depth, f)
                add(kind, "multiple-entailed", depth, depth, n_multiple)
        if kind == "which":
            # Required sole-entailment-with-contingent cases have no complete
            # membership-depth descriptor; they belong to the semantic cohort.
            add(kind, "A-D", 1, None, 24, contingent=True)
            for span, n in allocate(276, weights).items():
                add(kind, "A-D", int(span), None, n)
        for category in ("no-entailed-answer", "inconsistent"):
            for span, n in allocate(
                policy["training_allocation"][kind][category], weights
            ).items():
                add(kind, category, int(span), None, n)
    if len(slots) != 6000:
        raise ValueError(
            "replacement schedule must contain exactly 6000 requested slots"
        )
    validate_schedule(slots, policy)
    return tuple(slots)


def validate_schedule(slots, policy):
    if Counter(s.stratum for s in slots) != Counter(policy["strata"]):
        raise ValueError("scheduled evidence strata differ from preserved quotas")
    for kind, targets in policy["training_allocation"].items():
        if Counter(s.category for s in slots if s.kind == kind) != Counter(targets):
            raise ValueError("scheduled answer targets differ")
    counts = Counter(
        f"{s.stratum}/{s.count_value}"
        for s in slots
        if s.kind == "count" and s.depth is not None
    )
    if counts != Counter(policy["count_targets"]):
        raise ValueError("scheduled exact Count targets differ")


def validate_records(records, policy):
    """Fail closed on every numerical safeguard, not just the answer totals."""
    if len(records) != 6000 or len({r["query_key"] for r in records}) != 6000:
        raise ValueError("expected 6000 distinct canonical training questions")
    if len({r["base_id"] for r in records}) != 6000:
        raise ValueError("duplicate training base IDs")
    if Counter(r["stratum"] for r in records) != Counter(policy["strata"]):
        raise ValueError("selected evidence strata differ from preserved quotas")
    for kind, targets in policy["training_allocation"].items():
        actual = Counter(
            answer_category(r["rows"][0]) for r in records if r["query_kind"] == kind
        )
        if actual != Counter(targets):
            raise ValueError(f"{kind} answer/subtype counts differ")
    count_values = Counter(
        f"{r['stratum']}/{r['controlled_count_value']}"
        for r in records
        if "controlled_count_value" in r
    )
    if count_values != Counter(policy["count_targets"]):
        raise ValueError("selected Count quotas differ")
    for record in records:
        if (
            record["split"] != "train"
            or record["source"] != "controlled"
            or record["status"] != "accepted"
            or len(record["rows"]) != 1
            or record.get("independent_audit") != "passed"
            or record["rows"][0].get("selected_depth_audit") != "passed"
        ):
            raise ValueError("record lacks required independent admission")
        if len(record["tokens"]) != 2 or {t["style"] for t in record["tokens"]} != {
            "natural",
            "symbolic",
        }:
            raise ValueError("record lacks matched style token measurements")
        if not all(t["admitted"] for t in record["tokens"]):
            raise ValueError("token limit exceeded")
        depth = record["measurement"]["depth"]
        if depth is not None and not 1 <= depth <= 8:
            raise ValueError("training has out-of-range measured depth")
        if not record["stratum"].endswith("/semantic"):
            expected = int(record["stratum"].rsplit("/", 1)[1])
            if (
                depth != expected
                or not record["rows"][0]["selected_explanation_depth_support"]
            ):
                raise ValueError("depth stratum lacks selected explanation support")
    failures = []
    for stratum, minimum in policy["depth_family_minima"].items():
        families = {r["family_id"] for r in records if r["stratum"] == stratum}
        if len(families) < minimum:
            failures.append(
                {
                    "stratum": stratum,
                    "minimum_families": minimum,
                    "actual": len(families),
                }
            )
    for rule in policy["coverage_minima"]:
        found = [r for r in records if matches(r, rule)]
        families = {r["family_id"] for r in found}
        if len(found) < rule.get("minimum_bases", 0) or len(families) < rule.get(
            "minimum_families", 0
        ):
            failures.append(
                {
                    "rule": rule,
                    "actual_bases": len(found),
                    "actual_families": len(families),
                }
            )
    for kind in ("direction", "which", "count"):
        directions = {r["direction"] for r in records if r["query_kind"] == kind}
        if len(directions - {None}) != 8:
            failures.append({"query": kind, "reason": "missing direction coverage"})
    if failures:
        raise ValueError(f"preserved coverage requirements failed: {failures}")
