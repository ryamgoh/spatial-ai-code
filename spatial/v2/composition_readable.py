"""Readable paired evidence for the atomic composition fragment.

This renderer remains separate from production dataset generation.
"""

from dataclasses import replace
from itertools import combinations, pairwise

from spatial.v2.composition_compact import exclusion_groups
from spatial.v2.composition_conflicts import build_conflict, verify_conflict
from spatial.v2.composition_contracts import (
    UnsupportedEvidence,
    VerificationError,
    answer_domain,
    check_coordinates,
    encode,
    parse_answer,
    witness_outcome,
)
from spatial.v2.composition_evidence import CountSupport, verify_evidence
from spatial.v2.composition_scenarios import qualitative_outline, render_scenario
from spatial.v2.composition_shared_paths import collapsed_path, factor_shared_paths
from spatial.v2.difficulty import Axis, _axis_path, derive_axis_proof
from spatial.v2.grading import AnswerMode
from spatial.v2.model_certificates import ContingencyCertificate
from spatial.v2.solver import (
    And,
    Direction,
    DirectionQuery,
    SpatialSolverV2,
    WhichQuery,
    direction_constraint,
    direction_signs,
)

VERSION = "readable-shared-evidence-v4"
ABBREVIATIONS = {
    "North": "N",
    "Northeast": "NE",
    "East": "E",
    "Southeast": "SE",
    "South": "S",
    "Southwest": "SW",
    "West": "W",
    "Northwest": "NW",
}


def _path(problem, axis, nodes, indices, original_count=None):
    """Independently replay each displayed link directly against its cited atom."""
    if len(nodes) != len(indices) + 1 or not indices:
        raise UnsupportedEvidence("empty or malformed explanation path")
    if axis not in ("x", "y") or any(
        type(i) is not int or not 0 <= i < len(problem.premise.operands)
        for i in indices
    ):
        raise VerificationError("invalid axis or premise index")
    signs = []
    for (a, b), i in zip(pairwise(nodes), indices):
        atom = problem.premise.operands[i]
        sign = direction_signs(next(iter(atom.allowed)))["xy".index(axis)]
        if (a, b) == (atom.reference, atom.subject):
            sign = -sign
        elif (a, b) != (atom.subject, atom.reference):
            raise VerificationError("path cites a different object pair")
        signs.append(sign)
    if {-1, 1} <= set(signs):
        raise UnsupportedEvidence("a mixed-order path cannot be composed")
    # Increasing order is easier to read in both styles.
    if 1 in signs:
        nodes, indices, signs = nodes[::-1], indices[::-1], [-s for s in signs[::-1]]
    return {
        "kind": "path",
        "axis": axis,
        "nodes": list(nodes),
        "signs": signs,
        "indices": list(indices),
        "original_count": len(problem.premise.operands)
        if original_count is None
        else original_count,
    }


def _derive(problem, subject, reference, direction):
    proof = derive_axis_proof(problem, subject, reference, direction)
    if proof is None:
        raise UnsupportedEvidence("readable positive path is unavailable")
    paths = [
        _path(problem, p.axis.value, p.path, p.premise_indices)
        for p in (proof.x, proof.y)
    ]
    # Orient an equality chain to match the other axis when they share a route.
    x, y = paths
    if x["nodes"] == y["nodes"][::-1] and not any(x["signs"]):
        x["nodes"].reverse()
        x["indices"].reverse()
        x["signs"].reverse()
    return paths


def _opposing_path(problem, subject, reference, direction):
    choices = []
    for axis, required in zip((Axis.X, Axis.Y), direction_signs(direction)):
        for sign in (-1, 0, 1):
            if sign != required:
                path = _axis_path(problem, axis, subject, reference, sign)
                if path is not None:
                    choices.append(path)
    if not choices:
        return None
    chosen = min(choices, key=lambda p: (len(p.premise_indices), p.axis.value, p.path))
    return _path(problem, chosen.axis.value, chosen.path, chosen.premise_indices)


def _assumed_direction_conflict(problem, subject, reference, direction, solver):
    assumed = replace(
        problem,
        premise=And(
            (
                *problem.premise.operands,
                direction_constraint(subject, reference, direction),
            )
        ),
    )
    conflict = build_conflict(assumed)
    verify_conflict(conflict, solver)
    return [
        {
            "kind": "assume-direction",
            "subject": subject,
            "reference": reference,
            "direction": direction.value,
        },
        *_conflict(conflict, len(problem.premise.operands)),
    ]


def _direction_set(problem, possible, selected, witnesses, solver):
    units = [_model(problem, witnesses[v]) for v in selected if v in possible]
    groups = []
    for value in selected:
        if value in possible:
            continue
        direction = Direction(value)
        path = _opposing_path(
            problem, problem.query.target, problem.query.reference, direction
        )
        if path is None:
            units.extend(
                _assumed_direction_conflict(
                    problem,
                    problem.query.target,
                    problem.query.reference,
                    direction,
                    solver,
                )
            )
            units.append(
                {"kind": "excluded-directions", "values": [value], "basis": "conflict"}
            )
        else:
            # Directions excluded by the same comparison share its derivation.
            group = next((g for g in groups if g[0] == path), None)
            if group is None:
                groups.append((path, [value]))
            else:
                group[1].append(value)
    for path, values in groups:
        units.extend(
            (path, {"kind": "excluded-directions", "values": values, "basis": "axis"})
        )
    return units


def _model(problem, coordinates):
    check_coordinates(problem, coordinates)
    outcome = witness_outcome(problem, coordinates)
    query = problem.query
    return {
        "kind": "model",
        "points": [[obj, *coordinates[obj]] for obj in problem.objects],
        "query": "which"
        if isinstance(problem.query, WhichQuery)
        else "direction"
        if isinstance(problem.query, DirectionQuery)
        else "count",
        "outcome": sorted(outcome) if isinstance(outcome, frozenset) else outcome,
        "query_details": {
            "reference": query.reference,
            **(
                {"target": query.target}
                if isinstance(query, DirectionQuery)
                else {
                    "candidates": list(query.candidates),
                    "direction": next(iter(query.directions)).value,
                }
            ),
        },
    }


def _conflict(conflict, original_count=None):
    result = []
    for path in conflict.paths:
        result.append(
            _path(
                conflict.problem,
                "xy"[path[0].axis],
                (path[0].start, *(edge.end for edge in path)),
                tuple(edge.premise_index for edge in path),
                original_count,
            )
        )
    result.append(
        {
            "kind": "conflict",
            "reason": conflict.kind,
            "objects": [conflict.paths[0][0].start, conflict.paths[0][-1].end],
            "axis": "xy"[conflict.paths[0][0].axis],
        }
    )
    return result


def _membership(problem, item, solver):
    candidate = item.candidate
    direction = next(iter(problem.query.directions))
    status = item.status.value
    if status == "entailed":
        units = _derive(problem, candidate, problem.query.reference, direction)
    elif status == "impossible":
        path = _opposing_path(problem, candidate, problem.query.reference, direction)
        if path is None:
            units = _assumed_direction_conflict(
                problem, candidate, problem.query.reference, direction, solver
            )
            status = "impossible-by-conflict"
        else:
            units = [path]
    elif isinstance(item.evidence, ContingencyCertificate):
        units = [
            _model(model.problem, model.coordinates)
            for model in (item.evidence.witness, item.evidence.counterexample)
        ]
    else:
        raise UnsupportedEvidence("unsupported readable membership evidence")
    units.append(
        {
            "kind": "membership",
            "candidate": candidate,
            "status": status,
            "direction": direction.value,
            "reference": problem.query.reference,
        }
    )
    return units


def build_outline(evidence, menu, mode, *, solver=None):
    """Select shared factual units. Unsupported cases never fall back to transcripts."""
    solver = solver or SpatialSolverV2(backend="z3")
    verify_evidence(evidence, solver=solver)
    mode = AnswerMode(mode)
    semantics, certificate = evidence.semantics, evidence.certificate
    problem, possible = semantics.problem, semantics.possible
    decision = encode(semantics, menu, mode)
    domain = answer_domain(problem)
    selected = tuple(menu) if mode is AnswerMode.VISIBLE_POSSIBLE else domain
    units = []
    if not semantics.consistent:
        units = _conflict(certificate)
        reason = "inconsistent"
    elif decision.escape_reason == "incomplete-coverage":
        value = next(v for v in domain if v in possible and v not in menu)
        units = [_model(problem, semantics.witnesses[value])]
        reason = "unlisted"
    elif isinstance(problem.query, DirectionQuery):
        if semantics.case == "unique":
            direction = Direction(next(iter(possible)))
            units = _derive(
                problem, problem.query.target, problem.query.reference, direction
            )
            units.append(
                {
                    "kind": "direction",
                    "value": direction.value,
                    "target": problem.query.target,
                    "reference": problem.query.reference,
                    "excluded": exclusion_groups(direction),
                }
            )
            reason = "set"
        elif mode is AnswerMode.SINGLE:
            units = [
                _model(problem, semantics.witnesses[v])
                for v in tuple(v for v in domain if v in possible)[:2]
            ]
            reason = "different-outcomes"
        else:
            units = _direction_set(
                problem, possible, selected, semantics.witnesses, solver
            )
            reason = "set"
    elif isinstance(problem.query, WhichQuery):
        if len(problem.query.directions) != 1:
            raise UnsupportedEvidence("preview requires one membership direction")
        for item in certificate.candidates:
            if item.candidate in selected:
                units.extend(_membership(problem, item, solver))
        reason = semantics.case if mode is AnswerMode.SINGLE else "set"
    elif isinstance(certificate, CountSupport):
        if mode is AnswerMode.SINGLE and semantics.case == "underdetermined":
            units = [
                _model(problem, semantics.witnesses[v])
                for v in tuple(v for v in domain if v in possible)[:2]
            ]
            reason = "different-outcomes"
        else:
            if set(selected) - possible:
                required = tuple(certificate.memberships.entailed_entities)
                excluded = tuple(certificate.memberships.impossible_entities)
                for item in certificate.memberships.candidates:
                    if item.status.value != "contingent":
                        units.extend(
                            _membership(certificate.memberships.problem, item, solver)
                        )
                pairs = []
                for pair in certificate.incompatible_pairs:
                    units.append({"kind": "assume-pair", "members": list(pair.members)})
                    units.extend(
                        _conflict(pair.conflict, len(problem.premise.operands))
                    )
                    units.append(
                        {"kind": "exclude-pair", "members": list(pair.members)}
                    )
                    pairs.append(pair.members)
                allowed = set()
                for size in domain:
                    for members in combinations(problem.query.candidates, size):
                        group = set(members)
                        if (
                            set(required) <= group
                            and not set(excluded) & group
                            and not any(set(p) <= group for p in pairs)
                        ):
                            allowed.add(size)
                if allowed != set(possible):
                    raise UnsupportedEvidence(
                        "membership constraints do not characterize the possible counts"
                    )
                units.append(
                    {
                        "kind": "count-constraints",
                        "required": list(required),
                        "excluded": list(excluded),
                        "pairs": [list(p) for p in pairs],
                        "counts": sorted(allowed),
                        "rejected": [v for v in selected if v not in allowed],
                    }
                )
            units.extend(
                _model(problem, semantics.witnesses[v])
                for v in selected
                if v in possible
            )
            reason = "set"
    else:
        raise UnsupportedEvidence("unsupported preview certificate")
    values = [v for v in selected if v in possible]
    units.append(
        {
            "kind": "conclusion",
            "reason": reason,
            "values": values,
            "answer": decision.letters,
            "mode": mode.value,
            "unlisted": value if reason == "unlisted" else None,
        }
    )
    return factor_shared_paths(qualitative_outline(units, problem, VERSION))


def _join(values):
    items = list(map(str, values))
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _set(values):
    return "{" + ", ".join(map(str, values)) + "}"


def _render_path(unit, symbolic):
    if unit.get("shared_segments"):
        unit = collapsed_path(unit)
    axis, nodes, signs = unit["axis"], unit["nodes"], unit["signs"]
    original = unit["original_count"]
    citations = [
        i
        if isinstance(i, str)
        else f"p{i + 1}"
        if i < original
        else f"assumption {i - original + 1}"
        for i in unit["indices"]
    ]
    source = (
        ", ".join(citations)
        if symbolic
        else _join(
            i
            if isinstance(i, str)
            else str(i + 1)
            if i < original
            else f"temporary assumption {i - original + 1}"
            for i in unit["indices"]
        )
    )
    if symbolic:
        chain = f"{axis}({nodes[0]})" + "".join(
            f" {'<' if sign else '='} {axis}({node})"
            for sign, node in zip(signs, nodes[1:])
        )
        if len(signs) == 1:
            return f"[{source}] {chain}."
        return f"[{source}] {chain} => {axis}({nodes[0]}) {'<' if any(signs) else '='} {axis}({nodes[-1]})."
    cited = (
        f"({'sources' if unit.get('shared_segments') else 'premise sources'}: {source})"
    )
    if nodes[0] == nodes[-1] and any(signs):
        comparisons = [
            f"{a} {'has a smaller ' + axis + '-coordinate than' if sign else 'shares its ' + axis + '-coordinate with'} {b}"
            for (a, b), sign in zip(pairwise(nodes), signs)
        ]
        return (
            "The cited relations would require that "
            + "; ".join(comparisons)
            + f" {cited}."
        )
    if not any(signs):
        return f"{_join(nodes)} share the same {axis}-coordinate {cited}."
    if len(signs) == 1:
        return f"{nodes[-1]} has a greater {axis}-coordinate than {nodes[0]} {cited}."
    if all(signs):
        return (
            f"The {axis}-coordinates increase strictly along {_join(nodes)} {cited}. "
            f"So {nodes[-1]} has a greater {axis}-coordinate than {nodes[0]}."
        )
    comparisons = [
        f"{a} {'has a smaller ' + axis + '-coordinate than' if sign else 'shares its ' + axis + '-coordinate with'} {b}"
        for (a, b), sign in zip(pairwise(nodes), signs)
    ]
    return (
        f"Following the {axis}-coordinates, " + "; ".join(comparisons) + f" {cited}. "
        f"Therefore, {nodes[-1]} has a greater {axis}-coordinate than {nodes[0]}."
    )


def _render_unit(unit, symbolic):
    kind = unit["kind"]
    if kind == "path":
        prefix = f"Comparison {unit['comparison']}: " if "comparison" in unit else ""
        return prefix + _render_path(unit, symbolic)
    if kind == "direction":
        target, reference, value = unit["target"], unit["reference"], unit["value"]
        result = (
            f"=> {ABBREVIATIONS[value]}({target},{reference})."
            if symbolic
            else f"Together, these comparisons place {target} exactly {value.lower()} of {reference}."
        )
        for axis, values in unit["excluded"].items():
            if values:
                result += (
                    f"\n{axis}-comparison excludes {_set(ABBREVIATIONS[v] for v in values)}."
                    if symbolic
                    else f" The {axis}-comparison rules out {_join(v.lower() for v in values)}."
                )
        return result
    if kind == "scenario":
        return render_scenario(unit, symbolic, _join, ABBREVIATIONS)
    if kind == "membership":
        obj, status = unit["candidate"], unit["status"]
        if status == "contingent":
            first, second = unit["cases"]
            return (
                f"{obj} qualifies in Case {first}, not Case {second} => possible, not guaranteed."
                if symbolic
                else f"{obj} qualifies in Case {first} but not Case {second}. It is a possible match, not a guaranteed one."
            )
        if symbolic:
            return {
                "entailed": f"{ABBREVIATIONS[unit['direction']]}({obj},{unit['reference']}) in every valid map => {obj} always qualifies.",
                "impossible": f"Axis comparison contradicts {ABBREVIATIONS[unit['direction']]}({obj},{unit['reference']}) => {obj} cannot qualify.",
                "impossible-by-conflict": f"Assumed {ABBREVIATIONS[unit['direction']]}({obj},{unit['reference']}) leads to contradiction => {obj} cannot qualify.",
            }[status]
        return {
            "entailed": f"So {obj} is {unit['direction'].lower()} of {unit['reference']} in every valid map and always qualifies.",
            "impossible": f"This comparison is incompatible with {obj} being {unit['direction'].lower()} of {unit['reference']}, so it cannot qualify.",
            "impossible-by-conflict": f"Assuming {obj} is {unit['direction'].lower()} of {unit['reference']} leads to a contradiction, so it cannot qualify.",
        }[status]
    if kind == "conflict":
        a, b = unit["objects"]
        if unit["reason"] == "strict-cycle":
            return (
                f"{unit['axis']}({a}) < {unit['axis']}({a}) is impossible."
                if symbolic
                else "This would make a coordinate smaller than itself, which is impossible."
            )
        return (
            f"x({a})=x({b}) and y({a})=y({b}), but {a}!={b}: distinct-point contradiction."
            if symbolic
            else f"{a} and {b} would have both coordinates equal. They are distinct objects, so the no-coincidence rule makes this impossible."
        )
    if kind == "assume-pair":
        return f"{'Assume' if symbolic else 'Suppose'} {_join(unit['members'])} both qualify (temporary assumptions 1 and 2)."
    if kind == "assume-direction":
        subject, reference, direction = (
            unit["subject"],
            unit["reference"],
            unit["direction"],
        )
        return (
            f"Assume {ABBREVIATIONS[direction]}({subject},{reference}) (temporary assumption 1)."
            if symbolic
            else f"Suppose {subject} is exactly {direction.lower()} of {reference} (temporary assumption 1)."
        )
    if kind == "excluded-directions":
        basis = "This comparison" if unit["basis"] == "axis" else "This contradiction"
        return (
            f"{basis} excludes {_set(ABBREVIATIONS[v] for v in unit['values'])}."
            if symbolic
            else f"{basis} rules out {_join(v.lower() for v in unit['values'])}."
        )
    if kind == "exclude-pair":
        return f"Therefore, {_join(unit['members'])} cannot qualify together."
    if kind == "count-constraints":
        if symbolic:
            return (
                f"Required={_set(unit['required'])}; excluded={_set(unit['excluded'])}; "
                f"forbidden pairs={_set(_set(p) for p in unit['pairs'])}.\n"
                f"Joint membership subsets satisfying these constraints have sizes {_set(unit['counts'])}; "
                f"excluded counts={_set(unit['rejected'])}."
            )
        return (
            (
                f"Every qualifying group must include {_join(unit['required'])}. "
                if unit["required"]
                else "No object is required to qualify. "
            )
            + (
                f"{_join(unit['excluded'])} cannot qualify. "
                if unit["excluded"]
                else "No object is individually ruled out. "
            )
            + (
                "No qualifying group can contain both members of any forbidden pair: "
                + "; ".join(_join(p) for p in unit["pairs"])
                + ". "
                if unit["pairs"]
                else "There are no additional forbidden pairs. "
            )
            + f"Checking the eligible-object subsets against these constraints leaves only {'size' if len(unit['counts']) == 1 else 'sizes'} {_join(unit['counts'])}. "
            + (
                f"{'Count' if len(unit['rejected']) == 1 else 'Counts'} {_join(unit['rejected'])} {'is' if len(unit['rejected']) == 1 else 'are'} therefore impossible."
                if unit["rejected"]
                else ""
            )
        )
    if kind == "conclusion":
        reason, answer, mode = unit["reason"], unit["answer"], AnswerMode(unit["mode"])
        if reason == "inconsistent":
            text = "No valid map satisfies the premises. Every contract therefore requires E."
        elif reason == "unlisted":
            text = f"The witnessed value {unit['unlisted']} is absent from A-D, so the complete answer cannot be represented. Select F."
        elif reason == "different-outcomes":
            text = "These possibilities give different outcomes, so the answer is not uniquely determined. Select E."
        elif reason == "no-match":
            text = "No eligible object can qualify, so there is no unique qualifying object. Select E."
        elif reason == "multiple-certain-match":
            text = "Several objects always qualify, so there is no sole qualifying object. Select E."
        elif reason == "underdetermined":
            text = "The evidence does not establish one object as the only match in every valid map. Select E."
        else:
            domain = "listed" if mode is AnswerMode.VISIBLE_POSSIBLE else "complete"
            values = unit["values"]
            description = (
                f"The only {'listed ' if mode is AnswerMode.VISIBLE_POSSIBLE else ''}possible value is {values[0]}."
                if len(values) == 1
                else f"The {domain} possible values are {_join(values)}."
                if values
                else f"The {domain} possible set is empty."
            )
            text = (
                f"{domain.capitalize()} possible set = {_set(values)}."
                if symbolic
                else description
            )
            if answer == "F":
                text += (
                    " The unique value is unlisted, so select F."
                    if mode is AnswerMode.SINGLE
                    else " No displayed value qualifies; select F."
                )
            else:
                text += f" Select {answer}."
        return text + f"\nAnswer: {answer}"
    raise UnsupportedEvidence(f"unknown explanation unit: {kind}")


def render_outline(outline):
    if outline["version"] not in (VERSION, "readable-qualitative-evidence-v3"):
        raise VerificationError("unknown outline version")
    return {
        style: "\n\n".join(
            _render_unit(unit, style == "symbolic") for unit in outline["units"]
        )
        for style in ("natural", "symbolic")
    }


def check_pair(evidence, menu, mode, pair, *, solver=None):
    """Check trusted-template output integrity, not arbitrary free-form prose."""
    expected = render_outline(build_outline(evidence, menu, mode, solver=solver))
    if pair != expected:
        raise VerificationError("rendered text differs from the checked explanation")
    gold = encode(evidence.semantics, menu, mode).letters
    for text in pair.values():
        if parse_answer(text, mode, stage=1) != gold:
            raise VerificationError("readable explanation changed the answer")
