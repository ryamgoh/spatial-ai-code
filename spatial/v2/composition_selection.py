"""Coverage-protected selection from saved verified bundles; no generation."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from spatial.v2.composition_balance import KINDS, cell_key
from spatial.v2.composition_contracts import parse_prompt
from spatial.v2.composition_feasibility import stable_int
from spatial.v2.composition_training import digest
from spatial.v2.grading import AnswerMode

VERSION = "coverage-protected-bundles-v1"
SEED = 20261030


def features(row):
    prefix = (row["query_kind"], row["contract"])
    result = {("case", *prefix, row["semantic_case"])}
    if row["returned_count"] is not None:
        result.add(("size", *prefix, row["returned_count"]))
    else:
        result.add(("escape", *prefix, row["escape_reason"]))
    if row["query_kind"] == "count" and row["semantic_case"] != "inconsistent":
        result.add(("zero", *prefix, row["contains_zero"]))
    if row["substantive_depth_support"]:
        if row["depth"] is None or row["returned_count"] is None:
            raise ValueError("invalid substantive depth metadata")
        result.add(("depth", *prefix, row["depth"]))
    return result


def inventory(records):
    bundles = []
    base_ids = set()
    for record in records:
        if record["status"] != "accepted":
            continue
        base = record["base_id"]
        if base in base_ids:
            raise ValueError("duplicate base")
        base_ids.add(base)
        groups = defaultdict(list)
        for row in record["rows"]:
            if row["base_id"] != base or row["query_kind"] != record["query_kind"]:
                raise ValueError("row/base mismatch")
            groups[row["bundle_id"]].append(row)
        for bundle_id, rows in sorted(groups.items()):
            parsed = [parse_prompt(r["prompt"]) for r in rows]
            if (
                len(rows) != 3
                or {r["contract"] for r in rows} != {m.value for m in AnswerMode}
                or len({tuple(r["menu"]) for r in rows}) != 1
                or len({(p, menu) for p, menu, _ in parsed}) != 1
                or any(
                    menu != tuple(r["menu"]) or mode.value != r["contract"]
                    for r, (_, menu, mode) in zip(rows, parsed)
                )
                or any(set(r["references"]) != {"natural", "symbolic"} for r in rows)
            ):
                raise ValueError("incomplete or mismatched bundle")
            row = rows[0]
            topology = record["structure"]["core_topology"]
            bundles.append(
                {
                    "id": bundle_id,
                    "base": base,
                    "kind": record["query_kind"],
                    "family": record["family_id"],
                    "recipe": record["recipe"],
                    "topology": topology,
                    "rows": rows,
                    "stratum": (
                        record["query_kind"],
                        topology,
                        row["semantic_case"],
                        row["depth"],
                        row["possible_count"],
                        row["contains_zero"],
                    ),
                    "features": set().union(*(features(r) for r in rows)),
                    "cells": {cell_key(r) for r in rows},
                    "escapes": sum(r["answer"] in ("E", "F") for r in rows),
                }
            )
    if len({b["id"] for b in bundles}) != len(bundles):
        raise ValueError("duplicate bundle ID")
    return sorted(bundles, key=lambda b: b["id"])


def validate_selection(bundles, ids, quota, *, required_cells=(), recipe_caps=None):
    if len(ids) != len(set(ids)):
        raise ValueError("repeated selected ID")
    selected = [b for b in bundles if b["id"] in set(ids)]
    if len(selected) != len(ids):
        raise ValueError("unknown selected ID")
    if Counter(b["kind"] for b in selected) != Counter({k: quota for k in KINDS}):
        raise ValueError("task quota mismatch")
    if len({b["base"] for b in selected}) != len(selected):
        raise ValueError("multiple menus from a base")
    if len({b["stratum"] for b in selected}) != len(selected):
        raise ValueError("within-stratum topology cap exceeded")
    required = set().union(*(b["features"] for b in bundles))
    observed = set().union(*(b["features"] for b in selected))
    if required - observed:
        raise ValueError("protected coverage missing")
    if set(required_cells) - set().union(*(b["cells"] for b in selected)):
        raise ValueError("required fine-cell coverage missing")
    for (kind, recipe), cap in (recipe_caps or {}).items():
        if type(cap) is not int or cap < 0:
            raise ValueError("recipe caps must be nonnegative integers")
        if sum(b["kind"] == kind and b["recipe"] == recipe for b in selected) > cap:
            raise ValueError("recipe cap exceeded")


def select(bundles, quota=70, timeout_ms=120000):
    import z3

    opt = z3.Optimize()
    opt.set(timeout=timeout_ms)
    variables = [z3.Bool(f"bundle_{i}") for i in range(len(bundles))]
    groups = defaultdict(list)
    for i, bundle in enumerate(bundles):
        for key in (("base", bundle["base"]), ("stratum", bundle["stratum"])):
            groups[key].append(i)
    for group in groups.values():
        opt.add(z3.Sum([z3.If(variables[i], 1, 0) for i in group]) <= 1)
    for kind in KINDS:
        opt.add(
            z3.Sum(
                [
                    z3.If(v, 1, 0)
                    for v, b in zip(variables, bundles)
                    if b["kind"] == kind
                ]
            )
            == quota
        )
    feature_vars, cell_vars, topology_vars = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for v, b in zip(variables, bundles):
        for feature in b["features"]:
            feature_vars[feature].append(v)
        for cell in b["cells"]:
            cell_vars[cell].append(v)
        topology_vars[(b["kind"], b["topology"])].append(v)
    for key in sorted(feature_vars, key=repr):
        opt.add(z3.Or(feature_vars[key]))
    objectives = [
        opt.minimize(
            z3.Sum([z3.If(v, b["escapes"], 0) for v, b in zip(variables, bundles)])
        ),
        opt.maximize(
            z3.Sum([z3.If(z3.Or(cell_vars[k]), 1, 0) for k in sorted(cell_vars)])
        ),
        opt.maximize(
            z3.Sum(
                [z3.If(z3.Or(topology_vars[k]), 1, 0) for k in sorted(topology_vars)]
            )
        ),
    ]
    ranks = {
        b["id"]: rank
        for rank, b in enumerate(
            sorted(bundles, key=lambda b: (stable_int(f"{SEED}/{b['id']}"), b["id"]))
        )
    }
    objectives.append(
        opt.minimize(
            z3.Sum([z3.If(v, ranks[b["id"]], 0) for v, b in zip(variables, bundles)])
        )
    )
    status = opt.check()
    if status != z3.sat:
        return {
            "status": str(status),
            "reason": opt.reason_unknown()
            if status == z3.unknown
            else "hard constraints infeasible; no relaxation performed",
        }
    ids = [
        b["id"] for v, b in zip(variables, bundles) if z3.is_true(opt.model().eval(v))
    ]
    validate_selection(bundles, ids, quota)
    bounds = [{"lower": str(h.lower()), "upper": str(h.upper())} for h in objectives]
    return {
        "status": "optimal"
        if all(b["lower"] == b["upper"] for b in bounds)
        else "feasible-unproven",
        "objective_bounds": bounds,
        "bundle_ids": ids,
    }


def select_milp(
    bundles, quota=70, timeout_ms=120000, *, required_cells=(), recipe_caps=None
):
    """Same binary problem, with proven objectives fixed between MILP stages."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    n = len(bundles)
    constraints = []
    groups, feature_groups, cell_groups, topology_groups = (
        defaultdict(list) for _ in range(4)
    )
    for i, b in enumerate(bundles):
        groups[("base", b["base"])].append(i)
        groups[("stratum", b["stratum"])].append(i)
        for feature in b["features"]:
            feature_groups[feature].append(i)
        for cell in b["cells"]:
            cell_groups[cell].append(i)
        topology_groups[(b["kind"], b["topology"])].append(i)
    for indices in groups.values():
        constraints.append(({i: 1 for i in indices}, 0, 1))
    for kind in KINDS:
        constraints.append(
            ({i: 1 for i, b in enumerate(bundles) if b["kind"] == kind}, quota, quota)
        )
    for key in sorted(feature_groups, key=repr):
        constraints.append(({i: 1 for i in feature_groups[key]}, 1, np.inf))
    for cell in sorted(set(required_cells)):
        if cell not in cell_groups:
            return {"status": "unsat", "reason": "required fine cell absent from pool"}
        constraints.append(({i: 1 for i in cell_groups[cell]}, 1, np.inf))
    for (kind, recipe), cap in sorted((recipe_caps or {}).items()):
        if type(cap) is not int or cap < 0:
            raise ValueError("recipe caps must be nonnegative integers")
        constraints.append(
            (
                {
                    i: 1
                    for i, b in enumerate(bundles)
                    if b["kind"] == kind and b["recipe"] == recipe
                },
                0,
                cap,
            )
        )
    indicators = []
    for grouped in (cell_groups, topology_groups):
        variables = []
        for key in sorted(grouped):
            indices = grouped[key]
            y = n
            n += 1
            variables.append(y)
            # y is exactly OR(selected bundles belonging to this category).
            constraints.append(({**{i: 1 for i in indices}, y: -1}, 0, np.inf))
            for i in indices:
                constraints.append(({y: 1, i: -1}, 0, np.inf))
        indicators.append(variables)
    ranks = {
        b["id"]: rank
        for rank, b in enumerate(
            sorted(bundles, key=lambda b: (stable_int(f"{SEED}/{b['id']}"), b["id"]))
        )
    }
    objectives = [
        {i: b["escapes"] for i, b in enumerate(bundles)},
        {i: -1 for i in indicators[0]},
        {i: -1 for i in indicators[1]},
        {i: ranks[b["id"]] for i, b in enumerate(bundles)},
    ]
    values = []
    for stage, objective in enumerate(objectives):
        rr, cc, vv = [], [], []
        for row, (terms, _, _) in enumerate(constraints):
            for col, value in terms.items():
                rr.append(row)
                cc.append(col)
                vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(constraints), n)).tocsc()
        c = np.zeros(n)
        for i, value in objective.items():
            c[i] = value
        solution = milp(
            c,
            integrality=np.ones(n),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(
                matrix,
                [low for _, low, _ in constraints],
                [high for _, _, high in constraints],
            ),
            options={"time_limit": timeout_ms / 1000, "mip_rel_gap": 0.0},
        )
        if solution.status != 0:
            return {
                "status": "unsat" if solution.status == 2 else "unknown",
                "stage": stage,
                "reason": solution.message,
                "proven_objectives": values,
            }
        value = round(solution.fun)
        if abs(solution.fun - value) > 1e-5:
            raise ValueError("nonintegral objective")
        values.append(
            {
                "value": value,
                "dual_bound": float(solution.mip_dual_bound),
                "gap": float(solution.mip_gap),
            }
        )
        constraints.append((objective, value, value))
    ids = [b["id"] for i, b in enumerate(bundles) if solution.x[i] > 0.5]
    validate_selection(
        bundles, ids, quota, required_cells=required_cells, recipe_caps=recipe_caps
    )
    chosen = [b for b in bundles if b["id"] in set(ids)]
    measured = [
        sum(b["escapes"] for b in chosen),
        -len(set().union(*(b["cells"] for b in chosen))),
        -len({(b["kind"], b["topology"]) for b in chosen}),
        sum(ranks[b["id"]] for b in chosen),
    ]
    if measured != [v["value"] for v in values]:
        raise ValueError("independent objective check failed")
    return {"status": "optimal", "objective_bounds": values, "bundle_ids": ids}


def describe(bundles, ids):
    chosen = [b for b in bundles if b["id"] in set(ids)]
    rows = [r for b in chosen for r in b["rows"]]
    present = set().union(*(b["features"] for b in chosen))
    required = set().union(*(b["features"] for b in bundles))
    cells = set().union(*(b["cells"] for b in chosen))
    return {
        "bundles": len(chosen),
        "views": len(rows),
        "families": len({b["family"] for b in chosen}),
        "task_topologies": len({(b["kind"], b["topology"]) for b in chosen}),
        "fine_cells": len(cells),
        "fine_cells_missing": sorted(
            set().union(*(b["cells"] for b in bundles)) - cells
        ),
        "protected_features": len(present),
        "missing_protected_features": sorted(required - present, key=repr),
        "groups": {
            f"{kind}/{mode.value}": {
                "views": len(group),
                "escapes": sum(r["answer"] in ("E", "F") for r in group),
                "substantive": sum(r["returned_count"] is not None for r in group),
                "depths": dict(
                    sorted(
                        Counter(
                            r["depth"] for r in group if r["substantive_depth_support"]
                        ).items()
                    )
                ),
            }
            for kind in KINDS
            for mode in AnswerMode
            if (
                group := [
                    r
                    for r in rows
                    if r["query_kind"] == kind and r["contract"] == mode.value
                ]
            )
        },
        "recipes": dict(Counter(b["recipe"] for b in chosen)),
        "single_base_cells": sum(
            len({b["base"] for b in chosen if cell in b["cells"]}) == 1
            for cell in cells
        ),
    }


def report(result):
    before, after = result["before"], result.get("after")
    lines = [
        "# Coverage-Protected Selection Results",
        "",
        (
            "Existing 240-base pool only. No new sampling, explanations, training files, "
            "split assignment or seal. Original artifacts are unchanged."
        ),
        "",
        f"Solver result: **{result['selection']['status']}**.",
    ]
    if not after:
        return "\n".join(lines + [str(result["selection"])]) + "\n"
    lines += [
        "",
        "## Same-Size Comparison",
        "",
        "| Metric | Previous | Revised |",
        "| --- | ---: | ---: |",
        *[
            f"| {label} | {before[key]} | {after[key]} |"
            for key, label in (
                ("views", "Views per style"),
                ("fine_cells", "Fine-grained cells"),
                ("task_topologies", "Task/topology combinations"),
                ("families", "Source families"),
                ("single_base_cells", "Cells with only one base"),
            )
        ],
        (
            f"| Missing protected categories | {len(before['missing_protected_features'])} | "
            f"{len(after['missing_protected_features'])} |"
        ),
        "",
        "## Escape Rates",
        "",
        "| Task / contract | Previous E/F | Revised E/F | Revised substantive |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, group in after["groups"].items():
        old = before["groups"][key]
        lines.append(
            f"| {key} | {old['escapes']}/{old['views']} "
            f"({old['escapes'] / old['views']:.1%}) | "
            f"{group['escapes']}/{group['views']} ({group['escapes'] / group['views']:.1%}) | "
            f"{group['substantive']} |"
        )
    lines += [
        "",
        "## Substantive Unique Depths",
        "",
        "| Task | Depth | Pool bases | Previous | Revised |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for kind in KINDS:
        key = f"{kind}/single"
        for depth in range(1, 11):
            lines.append(
                f"| {kind} | {depth} | {result['pool_unique_depth'][kind].get(depth, 0)} | "
                f"{before['groups'][key]['depths'].get(depth, 0)} | "
                f"{after['groups'][key]['depths'].get(depth, 0)} |"
            )
    lines += [
        "",
        "## Pool Limits",
        "",
        "| Task | Available substantive Unique bases | Minimum E/F at 70 bases |",
        "| --- | ---: | ---: |",
        *[
            f"| {k} | {v} | {max(0, 70 - v)}/70 |"
            for k, v in result["pool_unique_bases"].items()
        ],
        "",
        (
            "These are availability bounds before protected coverage and topology caps, "
            "not promises that every available substantive base can coexist in the selected set."
        ),
        "",
        "## Interpretation",
        "",
        (
            "Hard minima preserve each observed semantic case, returned size, escape reason, "
            "Count-zero category and substantive depth by task/contract. They do not require "
            "every combination of these features or every literal answer permutation. The "
            "JSON lists missing fine-grained cells and all selected bundle IDs."
        ),
        "",
        (
            "The optimizer first minimizes total E/F under those hard constraints, then "
            "maximizes fine-cell coverage, then topology diversity, then a fixed seeded "
            "rank preference. Optimality applies only to this fixed pool and stated policy."
        ),
        "",
        (
            "Repeating a topology across different semantic/depth strata is allowed. Thus "
            "topology diversity can decrease relative to the previous blanket filter. "
            "Topology and source-family counts are not proofs of independent reasoning "
            "structures. No quota percentages, split budgets or independence thresholds "
            "have been approved for production."
        ),
        "",
        (
            "One example per protected category is a feasibility minimum, not adequate "
            "training or benchmark coverage. Selection cannot create unseen Exhaustive "
            "Direction size-four cases, invariant Count with varying identities, or "
            "additional generator families. The inherited certificate limitation is unchanged."
        ),
        "",
        (
            "All constraints are independently rechecked in Python after solving. Both "
            "styles and the three contracts are retained from each original bundle. "
            "No reference text, gold, depth or prompt is modified."
        ),
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("z3", "milp"), default="z3")
    args = parser.parse_args()
    md = args.output.with_suffix(".md")
    if args.output.exists() or md.exists():
        raise ValueError("refusing to overwrite results")
    manifest = json.loads((args.source / "manifest.json").read_text())
    hashes = {
        name: digest(args.source / name)
        for name in (
            "manifest.json",
            "attempts.jsonl",
            "summary.json",
            "cell-matrix.json",
        )
    }
    if any(
        digest(args.source / name) != value
        for name, value in manifest["output_sha256"].items()
    ):
        raise ValueError("source artifact hash mismatch")
    records = [
        json.loads(line)
        for line in (args.source / "attempts.jsonl").read_text().splitlines()
    ]
    old = json.loads((args.source / "summary.json").read_text())["diagnostic_selection"]
    bundles = inventory(records)
    selection = (select if args.backend == "z3" else select_milp)(bundles)
    result = {
        "version": VERSION,
        "backend": args.backend,
        "source_hashes": hashes,
        "selector_sha256": digest(Path(__file__)),
        "selection": selection,
        "before": describe(bundles, old["bundle_ids"]),
        "pool_unique_bases": {
            kind: len(
                {
                    b["base"]
                    for b in bundles
                    if b["kind"] == kind
                    and any(
                        r["contract"] == "single" and r["returned_count"] is not None
                        for r in b["rows"]
                    )
                }
            )
            for kind in KINDS
        },
        "pool_unique_depth": {
            kind: {
                d: len(
                    {
                        b["base"]
                        for b in bundles
                        if b["kind"] == kind
                        and any(
                            r["contract"] == "single"
                            and r["substantive_depth_support"]
                            and r["depth"] == d
                            for r in b["rows"]
                        )
                    }
                )
                for d in range(1, 11)
            }
            for kind in KINDS
        },
    }
    if "bundle_ids" in selection:
        result["after"] = describe(bundles, selection["bundle_ids"])
    if any(digest(args.source / name) != value for name, value in hashes.items()):
        raise ValueError("input changed during selection")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    md.write_text(report(result))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("before", "after", "selection")
            },
            indent=2,
        )
    )
    print(selection["status"])


if __name__ == "__main__":
    main()
