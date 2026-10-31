"""Coverage-constrained retention of verified bases from an ordered fixed reserve."""

from collections import Counter, defaultdict

from spatial.v2 import composition_production as historical
from spatial.v2.entailment_contracts import VERSION
from spatial.v2.solver import Direction

RETENTION = "coverage-constrained-v1"


def quota_targets(policy):
    common = {**policy, "schema": historical.SCHEMA}
    targets = {
        **{("stratum", k): n for k, n in historical.quotas(common).items()},
        **{("source", k): n for k, n in historical.source_quotas(common).items()},
        **{("count", k): n for k, n in historical.count_quotas(common).items()},
    }
    multi = policy.get("diagnostic_multi_selection_bases")
    if multi is not None:
        total = sum(
            n
            for (kind, key), n in targets.items()
            if kind == "stratum" and "/which/" in key
        )
        if type(multi) is not int or not 0 <= multi <= total:
            raise ValueError(
                "diagnostic multi-Selection target exceeds Selection bases"
            )
        targets[("selection-multi", "all-splits")] = multi
    return targets


def multi_selection(record):
    """Count semantic bases with 2-4 entailments, never views or E/F letters."""
    return record.get("query_kind") == "which" and any(
        row.get("menu_role") == "primary"
        and row.get("semantic_case") == "multiple-entailed"
        and 2 <= row.get("entailed_count", 0) <= 4
        for row in record["rows"]
    )


def quota_keys(record):
    keys = [
        ("stratum", record["stratum"]),
        ("source", record["source_stratum"]),
    ]
    if "controlled_count_value" in record:
        keys.append(
            ("count", f"{record['stratum']}/{record['controlled_count_value']}")
        )
    if multi_selection(record):
        keys.append(("selection-multi", "all-splits"))
    return keys


def requirements(policy):
    common = {**policy, "schema": historical.SCHEMA}
    return [
        *policy["coverage_minima"],
        *[
            {"record_match": {"stratum": cell}, "minimum_families": n}
            for cell, n in historical.depth_family_minima(common).items()
        ],
        *[
            {
                "record_match": {
                    "split": role,
                    "query_kind": kind,
                    "direction": direction.value,
                },
                "minimum_bases": 1,
            }
            for role in historical.ROLES
            for kind in historical.KINDS
            for direction in Direction
        ],
    ]


def matches(record, rule):
    if "record_match" in rule:
        return all(record.get(k) == v for k, v in rule["record_match"].items())
    return any(
        all(row.get(k) == v for k, v in rule["match"].items()) for row in record["rows"]
    )


def validate(candidates, selected, policy):
    ids = [r["base_id"] for r in selected]
    available = {r["base_id"] for r in candidates}
    if len(ids) != len(set(ids)) or not set(ids) <= available:
        raise ValueError("invalid retained base identities")
    targets = quota_targets(policy)
    counts = Counter(
        key
        for record in selected
        for key in quota_keys(record)
        if key[0] != "selection-multi" or key in targets
    )
    if counts != Counter(targets):
        raise ValueError("retained base/source/Count quotas do not match exactly")
    for rule in requirements(policy):
        found = [record for record in selected if matches(record, rule)]
        families = {historical.source_lineage(r["family_id"]) for r in found}
        if len(found) < rule.get("minimum_bases", 0) or len(families) < rule.get(
            "minimum_families", 0
        ):
            raise ValueError("retained coverage requirement is unsatisfied")


def select(candidates, policy):
    """Solve the selection problem only; spatial labels have already been checked."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    n = len(candidates)
    if not n or len({r["base_id"] for r in candidates}) != n:
        raise ValueError("retention requires distinct verified candidates")
    if any(
        r.get("interface") != VERSION or r["status"] != "accepted" for r in candidates
    ):
        raise ValueError("only verified entailment candidates may be retained")
    constraints = []
    groups = defaultdict(list)
    targets = quota_targets(policy)
    for i, record in enumerate(candidates):
        for key in quota_keys(record):
            if key[0] == "selection-multi" and key not in targets:
                continue
            groups[key].append(i)
    for key in sorted(set(targets) | set(groups)):
        constraints.append(
            ({i: 1 for i in groups[key]}, targets.get(key, 0), targets.get(key, 0))
        )
    for rule in requirements(policy):
        indices = [i for i, r in enumerate(candidates) if matches(r, rule)]
        if "minimum_bases" in rule:
            constraints.append(({i: 1 for i in indices}, rule["minimum_bases"], np.inf))
        if "minimum_families" in rule:
            families = defaultdict(list)
            for i in indices:
                families[historical.source_lineage(candidates[i]["family_id"])].append(
                    i
                )
            indicators = []
            for family in sorted(families):
                y = n
                n += 1
                indicators.append(y)
                constraints.append(
                    ({**{i: 1 for i in families[family]}, y: -1}, 0, np.inf)
                )
            constraints.append(
                ({i: 1 for i in indicators}, rule["minimum_families"], np.inf)
            )
    rr, cc, vv = [], [], []
    for row, (terms, _, _) in enumerate(constraints):
        for col, value in terms.items():
            rr.append(row)
            cc.append(col)
            vv.append(value)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(constraints), n)).tocsc()
    objective = np.zeros(n)
    objective[: len(candidates)] = np.arange(1, len(candidates) + 1)
    result = milp(
        objective,
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            matrix,
            [low for _, low, _ in constraints],
            [high for _, _, high in constraints],
        ),
        options={
            "time_limit": policy.get("retention_timeout_seconds", 60),
            "mip_rel_gap": 0.0,
        },
    )
    if result.status == 2:
        return None
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(
            f"retention solver did not establish a feasible subset: {result.message}"
        )
    selected = [r for i, r in enumerate(candidates) if result.x[i] > 0.5]
    validate(candidates, selected, policy)
    return {r["base_id"] for r in selected}


class CompletionGate:
    """Try retention at the first potentially feasible ordered prefix."""

    def __init__(self, policy):
        self.policy = policy
        self.candidates = []
        self.targets = quota_targets(policy)
        self.rules = requirements(policy)
        self.quota_supply = Counter()
        self.coverage_supply = Counter()
        self.family_supply = [set() for _ in self.rules]
        self.selected_ids = None
        self.solver_calls = 0
        self.verification_failed = False

    def satisfied(self, record, records):
        self.verification_failed |= record["status"] == "verification-failed"
        if record["status"] != "accepted":
            return False
        candidate = records[-1]
        self.candidates.append(candidate)
        if self.verification_failed:
            return False
        self.quota_supply.update(quota_keys(candidate))
        for index, rule in enumerate(self.rules):
            if matches(candidate, rule):
                self.coverage_supply[index] += 1
                self.family_supply[index].add(
                    historical.source_lineage(candidate["family_id"])
                )
        if any(self.quota_supply[key] < n for key, n in self.targets.items()):
            return False
        if any(
            self.coverage_supply[i] < rule.get("minimum_bases", 0)
            or len(self.family_supply[i]) < rule.get("minimum_families", 0)
            for i, rule in enumerate(self.rules)
        ):
            return False
        self.solver_calls += 1
        self.selected_ids = select(self.candidates, self.policy)
        return self.selected_ids is not None

    def diagnostic_ids(self):
        """A quota-only subset is diagnostic, never a coverage-passing release."""
        used, selected = Counter(), set()
        for record in self.candidates:
            keys = [
                key
                for key in quota_keys(record)
                if key[0] != "selection-multi" or key in self.targets
            ]
            if all(used[key] < self.targets.get(key, 0) for key in keys):
                used.update(keys)
                selected.add(record["base_id"])
        return selected
