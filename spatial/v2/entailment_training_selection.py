"""Select one verified menu per training question; evaluation remains fixed."""

from collections import Counter, defaultdict
from copy import deepcopy

from spatial.v2 import composition_production as historical
from spatial.v2 import entailment_retention as retention
from spatial.v2.entailment_unique_study import validate_row


def answer_category(row):
    if row["answer"] in ("A", "B", "C", "D"):
        return "A-D"
    reason = row["escape_reason"]
    if row["answer"] == "F" and reason == "unique-unlisted":
        return reason
    if row["answer"] == "E" and reason in (
        "multiple-entailed",
        "no-entailed-answer",
        "inconsistent",
    ):
        return reason
    raise ValueError("invalid Unique answer category")


def menu_record(base, row):
    result = {**base, "rows": [row]}
    result["tokens"] = [t for t in base["tokens"] if t["id"] == row["id"]]
    return result


def validate(candidates, selected, evaluation, policy):
    """Check exact menu membership, evidence markers and all inherited gates."""
    available = {r["base_id"]: r for r in candidates}
    if len(available) != len(candidates):
        raise ValueError("duplicate candidate bases")
    keys = set()
    for record in selected:
        if record["split"] != "train" or len(record["rows"]) != 1:
            raise ValueError("training requires exactly one menu per base")
        original = available[record["base_id"]]
        row = record["rows"][0]
        validate_row(row)
        if row not in original["rows"] or record != menu_record(original, row):
            raise ValueError("selected menu is not an unchanged candidate row")
        if record.get("independent_audit") != "passed":
            raise ValueError("candidate lacks independent audit")
        if row.get("selected_depth_audit") != "passed":
            raise ValueError("selected-menu evidence has not been replayed")
        if record["query_key"] in keys:
            raise ValueError("duplicate underlying training question")
        keys.add(record["query_key"])
        tokens = record["tokens"]
        if (
            len(tokens) != 2
            or {t["style"] for t in tokens} != {"natural", "symbolic"}
            or not all(t["admitted"] for t in tokens)
        ):
            raise ValueError("selected menu lacks matched admitted style records")
    for kind, target in policy["training_allocation"].items():
        actual = Counter(
            answer_category(r["rows"][0]) for r in selected if r["query_kind"] == kind
        )
        if actual != Counter(target):
            raise ValueError("training answer/subtype quotas differ")
    roles = defaultdict(set)
    for record in [*selected, *evaluation]:
        roles[historical.source_lineage(record["family_id"])].add(record["split"])
    if any(len(splits) > 1 for splits in roles.values()):
        raise ValueError("family crosses split assignments")
    if keys.intersection(r["query_key"] for r in evaluation):
        raise ValueError("training query overlaps evaluation")
    retention.validate([*candidates, *evaluation], [*selected, *evaluation], policy)


def select(candidates, evaluation, policy, *, seconds=120):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if any(r["split"] != "train" for r in candidates):
        raise ValueError("candidate pool must contain training bases only")
    if len({r["base_id"] for r in candidates}) != len(candidates):
        raise ValueError("candidate bases must be unique")
    if any(r["split"] == "train" for r in evaluation):
        raise ValueError("evaluation inventory must be held out")
    menus = [menu_record(base, row) for base in candidates for row in base["rows"]]
    if any(r["rows"][0].get("selected_depth_audit") != "passed" for r in menus):
        raise ValueError("all selectable menu depth evidence must be replayed")
    constraints, groups = [], defaultdict(list)
    n = len(menus)
    for i, record in enumerate(menus):
        groups[record["base_id"]].append(i)
    for indices in groups.values():
        constraints.append(({i: 1 for i in indices}, 0, 1))
    # At most one independent base for a canonical question, even if IDs differ.
    queries = defaultdict(list)
    for i, record in enumerate(menus):
        queries[record["query_key"]].append(i)
    for indices in queries.values():
        constraints.append(({i: 1 for i in indices}, 0, 1))
    targets = retention.quota_targets(policy)
    fixed = Counter(
        k for r in evaluation for k in retention.quota_keys(r) if k in targets
    )
    for key, target in targets.items():
        indices = [i for i, r in enumerate(menus) if key in retention.quota_keys(r)]
        needed = target - fixed[key]
        constraints.append(({i: 1 for i in indices}, needed, needed))
    for kind, categories in policy["training_allocation"].items():
        for category, target in categories.items():
            indices = [
                i
                for i, r in enumerate(menus)
                if r["query_kind"] == kind and answer_category(r["rows"][0]) == category
            ]
            constraints.append(({i: 1 for i in indices}, target, target))
    for rule in retention.requirements(policy):
        indices = [i for i, r in enumerate(menus) if retention.matches(r, rule)]
        fixed_rows = [r for r in evaluation if retention.matches(r, rule)]
        if "minimum_bases" in rule:
            constraints.append(
                (
                    {i: 1 for i in indices},
                    max(0, rule["minimum_bases"] - len(fixed_rows)),
                    np.inf,
                )
            )
        if "minimum_families" in rule:
            fixed_families = {
                historical.source_lineage(r["family_id"]) for r in fixed_rows
            }
            families = defaultdict(list)
            for i in indices:
                family = historical.source_lineage(menus[i]["family_id"])
                if family not in fixed_families:
                    families[family].append(i)
            indicators = []
            for family in sorted(families):
                indicator = n
                n += 1
                indicators.append(indicator)
                constraints.append(
                    (
                        {**{i: 1 for i in families[family]}, indicator: -1},
                        0,
                        np.inf,
                    )
                )
            constraints.append(
                (
                    {i: 1 for i in indicators},
                    max(0, rule["minimum_families"] - len(fixed_families)),
                    np.inf,
                )
            )
    rr, cc, vv = [], [], []
    for row, (terms, _, _) in enumerate(constraints):
        for col, value in terms.items():
            rr.append(row)
            cc.append(col)
            vv.append(value)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(constraints), n)).tocsc()
    objective = np.zeros(n)
    objective[: len(menus)] = np.arange(1, len(menus) + 1)
    result = milp(
        objective,
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            matrix, [lo for _, lo, _ in constraints], [hi for _, _, hi in constraints]
        ),
        options={"time_limit": seconds, "mip_rel_gap": 0.0},
    )
    if result.status == 2:
        return None
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(f"one-menu selection inconclusive: {result.message}")
    selected = [deepcopy(r) for i, r in enumerate(menus) if result.x[i] > 0.5]
    validate(candidates, selected, evaluation, policy)
    return selected
