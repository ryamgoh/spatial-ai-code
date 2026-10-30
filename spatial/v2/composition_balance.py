"""Bounded cell availability and bundle-balancing diagnostic, never a release."""

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from spatial.v2.composition_contracts import (
    Semantics,
    UnsupportedEvidence,
    analyze,
    answer_domain,
    encode,
    parse_prompt,
    render_prompt,
)
from spatial.v2.composition_dataset import random_stories
from spatial.v2.composition_depth import measure_depth
from spatial.v2.composition_evidence import build_evidence
from spatial.v2.composition_feasibility import (
    depth_story,
    graph_profile,
    scramble,
    stable_int,
    token_measurement,
    verify_transformations,
)
from spatial.v2.composition_pilot import construct_stories, story_key
from spatial.v2.composition_readable import (
    VERSION,
    build_outline,
    check_rendered_outline,
    render_outline,
)
from spatial.v2.composition_readable_audit import check_row
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.grading import AnswerMode
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    SpatialProblem,
    SpatialSolverV2,
    WhichQuery,
    direction_constraint,
    direction_signs,
)
from spatial.v2.structure import canonical_structure_profile

SEED = 20261030
KINDS = ("direction", "which", "count")


def schedule():
    rng = random.Random(SEED)
    entries = [
        (depth_story(d, sample, rng), d) for d in range(1, 11) for sample in range(3)
    ]
    entries.extend(
        (scramble(s, rng), None)
        for s in (
            *random_stories(SEED + 1, world_count=24, atom_count=16),
            *construct_stories(SEED + 2, 1),
        )
    )
    return tuple(
        (
            replace(
                s,
                story_id="balance-" + s.story_id,
                source_family="balance-" + s.source_family,
            ),
            d,
        )
        for s, d in entries
    )


def menu_inventory(semantics, seed):
    """One actual menu per feasible displayed-possible cardinality."""
    domain = answer_domain(semantics.problem)
    if len(domain) < 4:
        raise UnsupportedEvidence("fewer than four real menu values")
    yes = [v for v in domain if v in semantics.possible]
    no = [v for v in domain if v not in semantics.possible]
    rng = random.Random(seed)
    rng.shuffle(yes)
    rng.shuffle(no)
    menus = []
    for k in range(max(0, 4 - len(no)), min(4, len(yes)) + 1):
        menu = yes[:k] + no[: 4 - k]
        rng.shuffle(menu)
        menus.append(tuple(menu))
    return tuple(menus)


def topology_key(problem):
    """Label-free WL refinement on the query-connected graph, ignoring directions."""
    query = problem.query
    seeds = (
        {query.target, query.reference}
        if isinstance(query, DirectionQuery)
        else {query.reference, *query.candidates}
    )
    adjacent = {obj: set() for obj in problem.objects}
    for atom in problem.premise.operands:
        adjacent[atom.subject].add(atom.reference)
        adjacent[atom.reference].add(atom.subject)
    core = set(seeds)
    frontier = set(seeds)
    while frontier:
        frontier = {v for u in frontier for v in adjacent[u]} - core
        core.update(frontier)
    colors = {
        obj: "reference"
        if obj == query.reference
        else "target"
        if isinstance(query, DirectionQuery) and obj == query.target
        else "candidate"
        if not isinstance(query, DirectionQuery) and obj in query.candidates
        else "other"
        for obj in core
    }
    for _ in core:
        descriptors = {
            obj: (colors[obj], tuple(sorted(colors[n] for n in adjacent[obj])))
            for obj in core
        }
        palette = {
            value: str(i) for i, value in enumerate(sorted(set(descriptors.values())))
        }
        updated = {obj: palette[value] for obj, value in descriptors.items()}
        stable = len(set(updated.values())) == len(set(colors.values()))
        colors = updated
        if stable:
            break
    edges = sorted(
        tuple(sorted((colors[a], colors[b])))
        for a in core
        for b in adjacent[a]
        if a < b
    )
    return str(
        stable_int(json.dumps([type(query).__name__, sorted(colors.values()), edges]))
    )


def query_profile(problem):
    atoms = set(problem.premise.operands)
    for atom in tuple(atoms):
        signs = direction_signs(next(iter(atom.allowed)))
        inverse = next(
            d for d in Direction if direction_signs(d) == tuple(-s for s in signs)
        )
        atoms.add(direction_constraint(atom.reference, atom.subject, inverse))
    profile = canonical_structure_profile(
        replace(problem, premise=And(tuple(sorted(atoms, key=repr)))),
        max_permutations=720,
    )
    return {
        "key": str(stable_int(profile["canonical_problem"])),
        "method": profile["canonicalization"],
        "core_topology": topology_key(problem),
    }


def semantic_profile(semantics):
    return {
        "consistent": semantics.consistent,
        "case": semantics.case,
        "possible": sorted(semantics.possible),
        "certain": sorted(semantics.certain),
        "domain_size": len(answer_domain(semantics.problem)),
    }


def row_metadata(semantics, menu, mode):
    decision = encode(semantics, menu, mode)
    return {
        "contract": mode.value,
        "semantic_case": decision.semantic_case,
        "answer": decision.letters,
        "escape_reason": decision.escape_reason,
        "possible_count": len(semantics.possible),
        "displayed_possible_count": len(set(menu) & semantics.possible),
        "returned_count": None
        if decision.letters in ("E", "F")
        else len(decision.letters),
        "coverage": "full" if semantics.possible <= set(menu) else "partial",
        "contains_zero": isinstance(semantics.problem.query, CountQuery)
        and 0 in semantics.possible,
    }


CELL_FIELDS = (
    "query_kind",
    "contract",
    "semantic_case",
    "possible_count",
    "displayed_possible_count",
    "returned_count",
    "escape_reason",
    "contains_zero",
)


def cell_key(row):
    return json.dumps([row[field] for field in CELL_FIELDS])


def interface_matrix():
    """Enumerate abstract answer-set/menu states, NOT spatial realizability."""
    cells = {}
    for kind in KINDS:
        for n in (8,) if kind == "direction" else range(4, 9):
            objects = ("O", *(f"C{i}" for i in range(n)))
            query = (
                DirectionQuery("C0", "O")
                if kind == "direction"
                else (WhichQuery if kind == "which" else CountQuery)(
                    frozenset({Direction.NORTH}), "O", objects[1:]
                )
            )
            # Encoding only needs a legal domain shell; this atom is never solved
            # or offered as evidence for the abstract answer sets below.
            problem = SpatialProblem(
                objects, And((direction_constraint("C0", "O", Direction.NORTH),)), query
            )
            domain = answer_domain(problem)
            for size in range(len(domain) + 1):
                # Count needs both zero-present and zero-absent states.
                sets = [frozenset(domain[:size])]
                if kind == "count" and 0 < size < len(domain):
                    sets.append(frozenset(domain[1 : size + 1]))
                for possible in sets:
                    consistent_options = (
                        (False, True)
                        if size == 0 and kind == "which"
                        else ((False,) if size == 0 else (True,))
                    )
                    for consistent in consistent_options:
                        certain_sizes = (
                            range(size + 1)
                            if kind == "which"
                            else (size if size == 1 else 0,)
                        )
                        for certain_size in certain_sizes:
                            semantics = Semantics(
                                problem,
                                consistent,
                                possible,
                                frozenset(sorted(possible)[:certain_size]),
                                "abstract-interface",
                                {},
                            )
                            for menu in menu_inventory(semantics, SEED):
                                for mode in AnswerMode:
                                    row = {
                                        "query_kind": kind,
                                        **row_metadata(semantics, menu, mode),
                                    }
                                    key = cell_key(row)
                                    cell = cells.setdefault(
                                        key,
                                        {
                                            **{f: row[f] for f in CELL_FIELDS},
                                            "domain_sizes": set(),
                                            "literal_gold_potential": set(),
                                            "status": "interface-permitted-not-yet-spatially-witnessed",
                                        },
                                    )
                                    cell["domain_sizes"].add(len(domain))
                                    gold = row["answer"]
                                    cell["literal_gold_potential"].update(
                                        (gold,)
                                        if gold in ("E", "F")
                                        else (
                                            "".join(c)
                                            for c in combinations("ABCD", len(gold))
                                        )
                                    )
    return {
        key: {
            **value,
            "domain_sizes": sorted(value["domain_sizes"]),
            "literal_gold_potential": sorted(value["literal_gold_potential"]),
        }
        for key, value in cells.items()
    }


def build_attempt(
    story, problem, requested_depth, tokenizer, solver, *, menu_selector=menu_inventory
):
    kind = type(problem.query).__name__.removesuffix("Query").lower()
    base_id = story.story_id + "-" + kind
    record = {
        "base_id": base_id,
        "story_id": story.story_id,
        "family_id": story.source_family,
        "recipe": story.recipe,
        "query_kind": kind,
        "requested_depth": requested_depth,
        "problem": tagged_dataclass_to_dict(problem),
        "graph": graph_profile(story),
        "stage": "semantics",
    }
    try:
        semantics = analyze(problem, solver=solver, include_witnesses=False)
        record["semantics"] = semantic_profile(semantics)
        menus = menu_selector(semantics, stable_int(base_id))
        record["semantic_views"] = [
            {"query_kind": kind, **row_metadata(semantics, menu, mode)}
            for menu in menus
            for mode in AnswerMode
        ]
        record["stage"] = "certificates"
        evidence = build_evidence(problem, solver=solver)
        record["stage"] = "transformations"
        verify_transformations(story, problem, semantics, solver)
        record["stage"] = "depth"
        try:
            measured = measure_depth(problem, solver=solver)
        except UnsupportedEvidence as exc:
            measured = {"depth": None, "reason": str(exc)}
        record["measurement"] = measured
        if requested_depth is not None and measured["depth"] != requested_depth:
            raise UnsupportedEvidence(
                f"depth mismatch: {requested_depth} vs {measured['depth']}"
            )
        record["stage"] = "render-and-token-check"
        rows, tokens = [], []
        for index, menu in enumerate(menus):
            for mode in AnswerMode:
                prompt = render_prompt(problem, menu, mode)
                if parse_prompt(prompt) != (problem, menu, mode):
                    raise ValueError("prompt roundtrip mismatch")
                outline = build_outline(evidence, menu, mode, solver=solver)
                pair = render_outline(outline)
                check_rendered_outline(evidence, menu, mode, outline, pair)
                row = {
                    "id": f"{base_id}-m{index}-{mode.value}",
                    "base_id": base_id,
                    "bundle_id": f"{base_id}-m{index}",
                    "query_kind": kind,
                    "split": "balance-feasibility-only",
                    "menu": list(menu),
                    "depth": measured["depth"],
                    "prompt": prompt,
                    "references": pair,
                    "explanation_outline": outline,
                    **row_metadata(semantics, menu, mode),
                }
                row["substantive_depth_support"] = (
                    row["returned_count"] is not None
                    and measured["depth"] is not None
                    and any(
                        u["kind"] == "path" and len(u["indices"]) == measured["depth"]
                        for u in outline["units"]
                    )
                )
                check_row(row, record["problem"])
                rows.append(row)
                tokens.extend(
                    token_measurement(tokenizer, row, style) for style in pair
                )
        record.update(
            rows=rows,
            tokens=tokens,
            evidence=tagged_dataclass_to_dict(evidence),
            status="accepted"
            if all(t["admitted"] for t in tokens)
            else "token-rejected",
            stage="complete",
        )
    except (ValueError, RuntimeError) as exc:
        record.update(
            status="unsupported"
            if isinstance(exc, UnsupportedEvidence)
            else "verification-failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
    return record


def diagnostic_selection(records):
    """Select complete bundles; do not independently rebalance contract views."""
    choices = defaultdict(list)
    for record in records:
        if record["status"] != "accepted":
            continue
        bundles = defaultdict(list)
        for row in record["rows"]:
            bundles[row["bundle_id"]].append(row)
        best = min(
            bundles.values(),
            key=lambda rows: (
                sum(r["answer"] in ("E", "F") for r in rows),
                rows[0]["coverage"] != "full",
                stable_int(rows[0]["bundle_id"]),
            ),
        )
        choices[record["query_kind"]].append((record, best))
    retained = {}
    for kind in KINDS:
        seen, selected = set(), []
        for record, bundle in sorted(
            choices[kind], key=lambda v: stable_int(v[0]["base_id"])
        ):
            topology = record["structure"]["core_topology"]
            if topology not in seen:
                selected.append((record, bundle))
                seen.add(topology)
        retained[kind] = selected
    quota = min(map(len, retained.values()))
    chosen = [
        (record, rows) for kind in KINDS for record, rows in retained[kind][:quota]
    ]
    rows = [r for _, bundle in chosen for r in bundle]
    return {
        "status": "availability-diagnostic-not-training-selection",
        "bundles_per_question_type": quota,
        "available_distinct_core_topologies": {k: len(v) for k, v in retained.items()},
        "base_ids": [r["base_id"] for r, _ in chosen],
        "bundle_ids": [bundle[0]["bundle_id"] for _, bundle in chosen],
        "views": len(rows),
        "groups": dict(Counter(f"{r['query_kind']}/{r['contract']}" for r in rows)),
        "mix_by_contract": {
            mode.value: dict(
                Counter(
                    "escape"
                    if r["answer"] in ("E", "F")
                    else "substantive-depth"
                    if r["substantive_depth_support"]
                    else "substantive-unassigned"
                    for r in rows
                    if r["contract"] == mode.value
                )
            )
            for mode in AnswerMode
        },
        "depth_support": dict(
            Counter(
                f"{r['query_kind']}/{r['depth']}"
                for r in rows
                if r["contract"] == AnswerMode.SINGLE.value
                and r["substantive_depth_support"]
            )
        ),
    }


def summarize(records):
    matrix = interface_matrix()
    seen, admitted, topologies = defaultdict(set), defaultdict(set), defaultdict(set)
    for record in records:
        for row in record.get("semantic_views", []):
            seen[cell_key(row)].add(record["base_id"])
        if record["status"] == "accepted":
            for row in record["rows"]:
                key = cell_key(row)
                admitted[key].add(record["base_id"])
                topologies[key].add(record["structure"]["core_topology"])
    if set(seen) - set(matrix):
        raise ValueError("observed semantic cell absent from interface matrix")
    for key, cell in matrix.items():
        cell.update(
            spatial_bases=len(seen[key]),
            admitted_bases=len(admitted[key]),
            admitted_core_topologies=len(topologies[key]),
            status="renderer-admitted"
            if admitted[key]
            else "solver-witnessed-renderer-not-admitted"
            if seen[key]
            else "interface-permitted-unobserved-not-proven-impossible",
        )
    accepted = [r for r in records if r["status"] == "accepted"]
    rows = [row for r in accepted for row in r["rows"]]
    summary = {
        "attempted_bases": len(records),
        "statuses": dict(Counter(r["status"] for r in records)),
        "failures": [
            {k: r.get(k) for k in ("base_id", "recipe", "stage", "status", "reason")}
            for r in records
            if r["status"] != "accepted"
        ],
        "rows_per_style": len(rows),
        "bundles": len({r["bundle_id"] for r in rows}),
        "by_recipe": {
            recipe: {
                "attempts": len(group),
                "accepted": sum(r["status"] == "accepted" for r in group),
                "accepted_story_keys": len(
                    {r["story_key"] for r in group if r["status"] == "accepted"}
                ),
                "accepted_query_keys": len(
                    {r["structure"]["key"] for r in group if r["status"] == "accepted"}
                ),
                "accepted_core_topologies": len(
                    {
                        r["structure"]["core_topology"]
                        for r in group
                        if r["status"] == "accepted"
                    }
                ),
            }
            for recipe in sorted({r["recipe"] for r in records})
            if (group := [r for r in records if r["recipe"] == recipe])
        },
        "cell_statuses": dict(Counter(c["status"] for c in matrix.values())),
        "admitted_cells_with_one_base": sum(
            c["admitted_bases"] == 1 for c in matrix.values()
        ),
        "admitted_cells_with_one_core_topology": sum(
            c["admitted_core_topologies"] == 1 for c in matrix.values()
        ),
        "literal_gold_observed": {
            f"{kind}/{mode.value}": sorted(
                {
                    r["answer"]
                    for r in rows
                    if r["query_kind"] == kind and r["contract"] == mode.value
                }
            )
            for kind in KINDS
            for mode in AnswerMode
        },
        "controlled_depth": dict(
            Counter(
                f"{r['query_kind']}/{r['requested_depth']}"
                for r in accepted
                if r["requested_depth"] is not None
            )
        ),
        "null_depth_reasons": dict(
            Counter(
                r["measurement"].get("reason")
                for r in accepted
                if r["measurement"]["depth"] is None
            )
        ),
        "tokens": {
            style: {
                "max": max(
                    (
                        t["target_tokens"]
                        for r in records
                        for t in r.get("tokens", [])
                        if t["style"] == style
                    ),
                    default=0,
                ),
                "mean": sum(
                    t["target_tokens"]
                    for r in accepted
                    for t in r["tokens"]
                    if t["style"] == style
                )
                / max(1, len(rows)),
            }
            for style in ("natural", "symbolic")
        },
        "diagnostic_selection": diagnostic_selection(records),
    }
    return summary, list(matrix.values())


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.tokenizer.name != REVISION:
        raise ValueError("expected new output directory and pinned tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    solver = SpatialSolverV2(backend="z3", timeout_ms=5000)
    root = Path(__file__).resolve().parents[2]
    protocol = (
        root
        / "experiments/spatialentail-composition-v1/research-analysis/BALANCE-FEASIBILITY-PROTOCOL-v1.md"
    )
    sources = [*sorted((root / "spatial/v2").glob("*.py")), protocol]
    entries = schedule()
    manifest = {
        "status": "running-feasibility-only",
        "seed": SEED,
        "renderer": VERSION,
        "attempt_budget": len(entries) * 3,
        "stories": len(entries),
        "solver_timeout_ms": 5000,
        "source_sha256": {str(p.relative_to(root)): digest(p) for p in sources},
        "tokenizer_sha256": {
            p.name: digest(p) for p in sorted(args.tokenizer.iterdir()) if p.is_file()
        },
        "schedule": [
            {
                "story_id": s.story_id,
                "recipe": s.recipe,
                "requested_depth": d,
                "problem": tagged_dataclass_to_dict(s.problems()[0]),
            }
            for s, d in entries
        ],
    }
    args.output.mkdir()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "interface-matrix.json").write_text(
        json.dumps(list(interface_matrix().values()), indent=2) + "\n"
    )
    records = []
    with (args.output / "attempts.jsonl").open("w") as stream:
        for story, requested in entries:
            key = story_key(story.objects, story.premise)
            for problem in story.problems():
                record = build_attempt(story, problem, requested, tokenizer, solver)
                record.update(story_key=key, structure=query_profile(problem))
                records.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
            print(
                f"{len(records)}/{len(entries) * 3}: {dict(Counter(r['status'] for r in records))}",
                flush=True,
            )
    if any(digest(root / p) != h for p, h in manifest["source_sha256"].items()):
        raise ValueError("source changed during run")
    summary, matrix = summarize(records)
    for name, value in (("summary.json", summary), ("cell-matrix.json", matrix)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    manifest["status"] = "completed-feasibility-only-not-production"
    manifest["output_sha256"] = {
        p.name: digest(p)
        for p in sorted(args.output.iterdir())
        if p.name != "manifest.json"
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
