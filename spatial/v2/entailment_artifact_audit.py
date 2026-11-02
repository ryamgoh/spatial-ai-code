"""Independent atomic, selected-depth and rendered-evidence audits."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from itertools import combinations, pairwise

import z3

from spatial.v2.entailment_dataset import audit_row
from spatial.v2.entailment_readable_v3 import audit_plan
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.entailment_scene_protocol import parse_row
from spatial.v2.solver import And, CountQuery, Direction, DirectionQuery

SIGNS = {
    "North": (0, 1),
    "Northeast": (1, 1),
    "East": (1, 0),
    "Southeast": (1, -1),
    "South": (0, -1),
    "Southwest": (-1, -1),
    "West": (-1, 0),
    "Northwest": (-1, 1),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relation(subject, reference, signs):
    return z3.And(
        *[
            a == b if sign == 0 else a > b if sign == 1 else a < b
            for a, b, sign in zip(subject, reference, signs)
        ]
    )


def checked_sat(solver, claim=None):
    solver.push()
    if claim is not None:
        solver.add(claim)
    result = solver.check()
    solver.pop()
    require(result in (z3.sat, z3.unsat), "inconclusive independent SMT audit")
    return result == z3.sat


def independent_claims(problem):
    points = {name: z3.Ints(f"x{i} y{i}") for i, name in enumerate(problem.objects)}
    solver = z3.Solver()
    for a, b in combinations(points.values(), 2):
        solver.add(z3.Or(a[0] != b[0], a[1] != b[1]))
    for atom in problem.premise.operands:
        solver.add(
            relation(
                points[atom.subject],
                points[atom.reference],
                SIGNS[next(iter(atom.allowed)).value],
            )
        )
    query = problem.query
    if isinstance(query, DirectionQuery):
        claims = {
            value: relation(points[query.target], points[query.reference], signs)
            for value, signs in SIGNS.items()
        }
    else:
        memberships = {
            name: relation(
                points[name],
                points[query.reference],
                SIGNS[next(iter(query.directions)).value],
            )
            for name in query.candidates
        }
        total = z3.Sum([z3.If(claim, 1, 0) for claim in memberships.values()])
        claims = (
            {value: total == value for value in range(len(query.candidates) + 1)}
            if isinstance(query, CountQuery)
            else memberships
        )
    return solver, claims


def replay_paths(problem, paths, subject, reference, direction, positive):
    axes = set()
    for path in paths:
        axis = "xy".index(path["axis"])
        nodes, indices = path["nodes"], path["indices"]
        require(len(nodes) == len(indices) + 1 and bool(indices), "malformed path")
        signs = []
        for (a, b), index in zip(pairwise(nodes), indices):
            require(
                type(index) is int and 0 <= index < len(problem.premise.operands),
                "invalid proof citation",
            )
            atom = problem.premise.operands[index]
            sign = SIGNS[next(iter(atom.allowed)).value][axis]
            if (a, b) == (atom.reference, atom.subject):
                sign = -sign
            else:
                require((a, b) == (atom.subject, atom.reference), "wrong proof edge")
            signs.append(sign)
        require(
            signs == path["signs"] and not ({-1, 1} <= set(signs)),
            "invalid composed proof signs",
        )
        derived = next((s for s in signs if s), 0)
        if (nodes[0], nodes[-1]) == (reference, subject):
            derived = -derived
        else:
            require((nodes[0], nodes[-1]) == (subject, reference), "wrong endpoints")
        require(
            (derived == SIGNS[direction][axis]) == positive,
            "proof does not establish membership status",
        )
        axes.add(path["axis"])
    require(axes == {"x", "y"} if positive else bool(axes), "incomplete proof axes")


def replay_counterexample(problem, value, points):
    require(set(points) == set(problem.objects), "incomplete counterexample")
    require(
        all(len(p) == 2 and all(type(v) is int for v in p) for p in points.values()),
        "invalid coordinates",
    )
    require(len(set(points.values())) == len(points), "coincident named objects")

    def signs(a, b):
        return tuple((x > y) - (x < y) for x, y in zip(points[a], points[b]))

    for atom in problem.premise.operands:
        require(
            signs(atom.subject, atom.reference)
            == SIGNS[next(iter(atom.allowed)).value],
            "counterexample violates premise",
        )
    query = problem.query
    if isinstance(query, DirectionQuery):
        holds = signs(query.target, query.reference) == SIGNS[value]
    else:
        members = {
            name
            for name in query.candidates
            if signs(name, query.reference) == SIGNS[next(iter(query.directions)).value]
        }
        holds = (
            len(members) == value if isinstance(query, CountQuery) else value in members
        )
    require(not holds, "counterexample fails to refute claim")


def label(consistent, entailed, menu, contract):
    if not consistent:
        return "E"
    visible = "".join(
        letter for letter, value in zip("ABCD", menu) if value in entailed
    )
    if contract == "unique-entailed":
        return visible or "F" if len(entailed) == 1 else "E"
    if contract == "listed-entailed":
        return visible or "F"
    require(contract == "exhaustive-entailed", "unknown contract")
    return visible if entailed and entailed <= set(menu) else "F"


def replay_conflict(problem, conflict):
    require(
        conflict is not None and conflict.problem == problem,
        "missing inconsistency proof",
    )
    cited = set()
    for path in conflict.paths:
        require(
            bool(path) and len({edge.axis for edge in path}) == 1,
            "malformed conflict path",
        )
        for edge in path:
            require(
                edge.axis in (0, 1) and type(edge.strict) is bool,
                "invalid conflict axis/status",
            )
            require(
                0 <= edge.premise_index < len(problem.premise.operands),
                "invalid conflict citation",
            )
            atom = problem.premise.operands[edge.premise_index]
            sign = SIGNS[next(iter(atom.allowed)).value][edge.axis]
            pair = (atom.subject, atom.reference)
            expected = pair if sign < 0 else pair[::-1]
            require(
                (edge.start, edge.end) in (pair, pair[::-1])
                if sign == 0
                else (edge.start, edge.end) == expected,
                "wrong conflict edge",
            )
            require(edge.strict == (sign != 0), "wrong conflict strictness")
            cited.add(edge.premise_index)
        require(
            all(a.end == b.start for a, b in pairwise(path)), "disconnected conflict"
        )
    if conflict.kind == "strict-cycle":
        require(len(conflict.paths) == 1, "invalid cycle path count")
        path = conflict.paths[0]
        require(
            path[0].start == path[-1].end and any(e.strict for e in path),
            "conflict is not a strict cycle",
        )
    else:
        require(
            conflict.kind == "forced-coincidence" and len(conflict.paths) == 2,
            "unknown conflict type",
        )
        x, y = conflict.paths
        require(
            x[0].axis == 0
            and y[0].axis == 1
            and (x[0].start, x[-1].end) == (y[0].start, y[-1].end)
            and x[0].start != x[-1].end
            and not any(e.strict for path in conflict.paths for e in path),
            "conflict does not force coincidence",
        )
    subset = replace(
        problem, premise=And(tuple(problem.premise.operands[i] for i in sorted(cited)))
    )
    solver, _ = independent_claims(subset)
    require(not checked_sat(solver), "cited conflict subset is satisfiable")


def audit_record(record):
    evidence = load_evidence(record)
    problem, semantics = evidence.semantics.problem, evidence.semantics
    solver, claims = independent_claims(problem)
    consistent = checked_sat(solver)
    entailed = {
        v for v, claim in claims.items() if not checked_sat(solver, z3.Not(claim))
    }
    if not consistent:
        entailed = set()
    require(
        (consistent, entailed) == (semantics.consistent, set(semantics.entailed)),
        "independent entailment mismatch",
    )
    if not consistent:
        replay_conflict(problem, evidence.conflict)
    else:
        require(evidence.conflict is None, "consistent case has conflict evidence")
    require(
        set(record["semantics"]["entailed"]) == entailed, "record/evidence mismatch"
    )
    negatives = set(claims) - entailed if consistent else set()
    require(set(semantics.counterexamples) == negatives, "missing negative evidence")
    for value, points in semantics.counterexamples.items():
        replay_counterexample(problem, value, points)
    require(set(evidence.positive_support) == entailed, "missing positive evidence")
    for value, support in evidence.positive_support.items():
        if isinstance(problem.query, CountQuery):
            items = support["memberships"]
            require(support["kind"] == "fixed-count", "unsupported Count evidence")
            require(
                [i["candidate"] for i in items] == list(problem.query.candidates),
                "Count proof omits candidate",
            )
            for item in items:
                require(type(item["qualifies"]) is bool, "invalid Count membership")
                replay_paths(
                    problem,
                    item["paths"],
                    item["candidate"],
                    problem.query.reference,
                    next(iter(problem.query.directions)).value,
                    item["qualifies"],
                )
            require(sum(i["qualifies"] for i in items) == value, "wrong joint Count")
        else:
            subject = (
                problem.query.target
                if isinstance(problem.query, DirectionQuery)
                else value
            )
            direction = (
                value
                if isinstance(problem.query, DirectionQuery)
                else next(iter(problem.query.directions)).value
            )
            replay_paths(
                problem,
                support["paths"],
                subject,
                problem.query.reference,
                direction,
                True,
            )
    categories = {
        value: "contingent" if checked_sat(solver, claims[value]) else "impossible"
        for value in negatives
    }
    bundles = defaultdict(list)
    for row in record["rows"]:
        parsed, menu, contract = parse_row(row)
        require(
            parsed == problem
            and list(menu) == row["menu"]
            and contract.value == row["contract"],
            "prompt differs from record",
        )
        expected = label(consistent, entailed, menu, row["contract"])
        require(row["answer"] == expected, "independent gold mismatch")
        require(
            row["negative_menu_categories"]
            == {
                letter: categories[v]
                for letter, v in zip("ABCD", menu)
                if v in categories
            },
            "negative metadata mismatch",
        )
        require(set(row["references"]) == {"natural", "symbolic"}, "unmatched styles")
        require(
            all(
                text.endswith(f"Answer: {expected}")
                for text in row["references"].values()
            ),
            "reference answer mismatch",
        )
        bundles[row["bundle_id"]].append(row)
    from spatial.v2.entailment_unique_study import STUDY_SCOPE, validate_row

    unique_only = record.get("study_scope") == STUDY_SCOPE
    if unique_only:
        for row in record["rows"]:
            validate_row(row)
    else:
        require(
            record.get("study_scope") is None
            and all(row.get("study_scope") is None for row in record["rows"]),
            "unknown or mismatched study scope",
        )
    require(
        all(
            len(rows) == (1 if unique_only else 3)
            and len({tuple(r["menu"]) for r in rows}) == 1
            and len({r["contract"] for r in rows}) == (1 if unique_only else 3)
            for rows in bundles.values()
        ),
        "unmatched contract bundle",
    )
    return len(entailed), len(negatives)


def certificate_depth(support):
    paths = (
        [path for member in support["memberships"] for path in member["paths"]]
        if support["kind"] == "fixed-count"
        else support["paths"]
    )
    require(bool(paths), "empty selected certificate")
    return max(len(path["indices"]) for path in paths)


def replay(record):
    """Return an annotated copy; never promote compact metadata without evidence."""
    audit_record(record)
    evidence = load_evidence(record)
    result = deepcopy(record)
    problem = evidence.semantics.problem
    # Reconstruct the renderer's complete-domain ordering independently.
    domain = (
        [d.value for d in Direction]
        if isinstance(problem.query, DirectionQuery)
        else list(range(len(problem.query.candidates) + 1))
        if isinstance(problem.query, CountQuery)
        else list(problem.query.candidates)
    )
    positives = evidence.positive_support
    values = [v for v in domain if v in positives]
    for row in result["rows"]:
        require(row["contract"] == "unique-entailed", "retired contract in depth audit")
        _, menu, _ = parse_row(row)
        actual = None
        if evidence.semantics.consistent and len(values) == 1:
            actual = certificate_depth(positives[values[0]])
            expected = "ABCD"[menu.index(values[0])] if values[0] in menu else "F"
            require(row["answer"] == expected, "sole answer/menu mismatch")
        elif evidence.semantics.consistent and len(values) > 1:
            cheapest = sorted(certificate_depth(s) for s in positives.values())[1]
            emitted = max(certificate_depth(positives[v]) for v in values[:2])
            if emitted == cheapest:
                actual = emitted
        if "selected_explanation_depth" in row:
            require(
                row["selected_explanation_depth"] == actual, "selected depth mismatch"
            )
        row["selected_explanation_depth"] = actual
        row["selected_explanation_depth_support"] = (
            actual is not None and actual == row["depth"]
        )
        # Verify that the visible reference text is exactly the evidence-derived
        # renderer output, not merely a correct final answer.
        audit_row(row, evidence)
        row["selected_depth_audit"] = "passed"
    result["independent_audit"] = "passed"
    return result


def independent_plan_check(evidence, plan):
    problem = evidence.semantics.problem
    for value in plan["values"]:
        support = evidence.positive_support[value]
        if isinstance(problem.query, CountQuery):
            for member in support["memberships"]:
                replay_paths(
                    problem,
                    member["paths"],
                    member["candidate"],
                    problem.query.reference,
                    next(iter(problem.query.directions)).value,
                    member["qualifies"],
                )
        else:
            replay_paths(
                problem,
                support["paths"],
                problem.query.target
                if isinstance(problem.query, DirectionQuery)
                else value,
                problem.query.reference,
                value
                if isinstance(problem.query, DirectionQuery)
                else next(iter(problem.query.directions)).value,
                True,
            )
    for value in plan["witness_keys"]:
        replay_counterexample(problem, value, evidence.semantics.counterexamples[value])
    audit_plan(evidence, plan)
