"""Evidence-preserving spatial prose and shared-chain rendering for Unique."""

from collections import defaultdict
from itertools import pairwise

from spatial.v2.composition_contracts import (
    VerificationError,
    answer_domain,
    witness_outcome,
)
from spatial.v2.composition_readable import _model
from spatial.v2.composition_scenarios import qualitative_outline, render_scenario
from spatial.v2.entailment_contracts import Contract, encode
from spatial.v2.entailment_evidence import ABBREVIATIONS, _join, _premise_key
from spatial.v2.solver import CountQuery, DirectionQuery, WhichQuery

VERSION = "spatial-prose-shared-chains-v3"


def _key(path):
    return (
        path["axis"],
        tuple(path["nodes"]),
        tuple(path["indices"]),
        tuple(path["signs"]),
    )


def contains_path(long, short):
    """Only exact, cited contiguous subpaths can reuse a displayed chain."""
    if long["axis"] != short["axis"]:
        return False
    orientations = [short]
    if not any(short["signs"]):
        orientations.append(
            {
                **short,
                "nodes": short["nodes"][::-1],
                "indices": short["indices"][::-1],
                "signs": short["signs"][::-1],
            }
        )
    for candidate in orientations:
        length = len(candidate["indices"])
        for start in range(len(long["indices"]) - length + 1):
            if (
                long["nodes"][start : start + length + 1] == candidate["nodes"]
                and long["indices"][start : start + length] == candidate["indices"]
                and long["signs"][start : start + length] == candidate["signs"]
            ):
                return True
    return False


def make_plan(evidence):
    semantics, problem = evidence.semantics, evidence.semantics.problem
    domain = answer_domain(problem)
    values = [v for v in domain if v in semantics.entailed][:2]
    paths = []
    members = []
    if semantics.consistent:
        for value in values:
            support = evidence.positive_support[value]
            if isinstance(problem.query, CountQuery):
                members = support["memberships"]
                paths.extend(p for member in members for p in member["paths"])
            else:
                paths.extend(support["paths"])
    unique = list({_key(p): p for p in paths}.values())
    chains = sorted(
        [
            p
            for p in unique
            if len(p["indices"]) > 1
            and not any(
                len(q["indices"]) > len(p["indices"]) and contains_path(q, p)
                for q in unique
            )
        ],
        key=lambda p: (-len(p["indices"]), _key(p)),
    )
    witness_keys = []
    outcomes = []
    exact_ambiguity = (
        semantics.consistent
        and not values
        and isinstance(problem.query, (DirectionQuery, CountQuery))
    )
    if exact_ambiguity:
        for value in domain:
            outcome = witness_outcome(problem, semantics.counterexamples[value])
            if outcome not in outcomes:
                witness_keys.append(value)
                outcomes.append(outcome)
            if len(outcomes) == 2:
                break
        if len(outcomes) != 2:
            raise VerificationError(
                "exact ambiguity requires two different checked outcomes"
            )
    elif (
        semantics.consistent
        and isinstance(problem.query, WhichQuery)
        and len(values) < 2
    ):
        witness_keys = [v for v in domain if v not in semantics.entailed]
    outline = qualitative_outline(
        [_model(problem, semantics.counterexamples[v]) for v in witness_keys],
        problem,
        VERSION,
    )
    groups = defaultdict(list)
    if isinstance(problem.query, WhichQuery):
        for value, witness in zip(
            witness_keys, outline["audit_witnesses"], strict=True
        ):
            groups[witness["case"]].append(value)
    return {
        "values": values,
        "paths": paths,
        "members": members,
        "chains": chains,
        "witness_keys": witness_keys,
        "outcomes": outcomes,
        "outline": outline,
        "negative_groups": dict(groups),
        "exact_ambiguity": exact_ambiguity,
    }


def audit_plan(evidence, plan):
    """Check every evidence obligation that grouping or witness selection can hide."""
    semantics, problem = evidence.semantics, evidence.semantics.problem
    expected_ambiguity = (
        semantics.consistent
        and not semantics.entailed
        and isinstance(problem.query, (DirectionQuery, CountQuery))
    )
    if plan["exact_ambiguity"] != expected_ambiguity:
        raise VerificationError("renderer changed the ambiguity proof obligation")
    expected_values = [v for v in answer_domain(problem) if v in semantics.entailed][:2]
    if plan["values"] != expected_values:
        raise VerificationError("renderer changed the selected positive certificate")
    expected_paths = []
    expected_members = []
    for value in expected_values:
        support = evidence.positive_support[value]
        if isinstance(problem.query, CountQuery):
            expected_members = support["memberships"]
            expected_paths.extend(p for m in expected_members for p in m["paths"])
        else:
            expected_paths.extend(support["paths"])
    if plan["paths"] != expected_paths or plan["members"] != expected_members:
        raise VerificationError(
            "renderer omitted or changed required membership/path evidence"
        )
    if any(_key(p) not in {_key(q) for q in expected_paths} for p in plan["chains"]):
        raise VerificationError("displayed chain is not an original certificate")
    for path in expected_paths:
        if len(path["indices"]) > 1 and not any(
            contains_path(c, path) for c in plan["chains"]
        ):
            raise VerificationError("required path is not covered by a displayed chain")
    if expected_paths and max(len(p["indices"]) for p in expected_paths) != max(
        [1, *(len(c["indices"]) for c in plan["chains"])]
    ):
        raise VerificationError("displayed chains lost the selected proof depth")
    expected_outline = qualitative_outline(
        [_model(problem, semantics.counterexamples[v]) for v in plan["witness_keys"]],
        problem,
        VERSION,
    )
    if plan["outline"] != expected_outline:
        raise VerificationError("displayed cases differ from saved counterexamples")
    if plan["exact_ambiguity"]:
        outcomes = [
            witness_outcome(problem, semantics.counterexamples[v])
            for v in plan["witness_keys"]
        ]
        if (
            len(outcomes) != 2
            or len(set(outcomes)) != 2
            or outcomes != plan["outcomes"]
        ):
            raise VerificationError(
                "ambiguity cases do not have different exact outcomes"
            )
    elif (
        semantics.consistent
        and isinstance(problem.query, WhichQuery)
        and len(expected_values) < 2
    ):
        covered = [v for values in plan["negative_groups"].values() for v in values]
        expected = set(answer_domain(problem)) - semantics.entailed
        if len(covered) != len(set(covered)) or set(covered) != expected:
            raise VerificationError("grouped Which explanation omits an alternative")
        for case, values in plan["negative_groups"].items():
            witnesses = [
                w for w in plan["outline"]["audit_witnesses"] if w["case"] == case
            ]
            if not witnesses:
                raise VerificationError("rejection group has no displayed case")
            points = {name: (x, y) for name, x, y in witnesses[0]["points"]}
            if set(values) & witness_outcome(problem, points):
                raise VerificationError(
                    "grouped case does not falsify every stated alternative"
                )


def _citations(path):
    return ", ".join(f"P{i + 1}" for i in path["indices"])


def _order_text(path, symbolic):
    axis, nodes, signs = path["axis"], path["nodes"], path["signs"]
    citations = _citations(path)
    if symbolic:
        return (
            f"[{citations}] {axis}({nodes[0]})"
            + "".join(
                f" {'<' if sign else '='} {axis}({node})"
                for sign, node in zip(signs, nodes[1:])
            )
            + "."
        )
    line = "vertical" if axis == "x" else "horizontal"
    lower = "west" if axis == "x" else "south"
    upper = "east" if axis == "x" else "north"
    if not any(signs):
        return f"{_join(nodes)} lie on the same {line} line ({citations})."
    if all(signs):
        return f"From {lower} to {upper}: {_join(nodes)} ({citations})."
    fragments = [
        f"{a} lies further {lower} than {b}"
        if sign
        else f"{a} and {b} lie on the same {line} line"
        for (a, b), sign in zip(pairwise(nodes), signs)
    ]
    return "; ".join(fragments) + f" ({citations})."


def _relation_text(problem, value, support, symbolic):
    query = problem.query
    target = query.target if isinstance(query, DirectionQuery) else value
    direction = (
        value
        if isinstance(query, DirectionQuery)
        else next(iter(query.directions)).value
    )
    paths = support["paths"]
    direct = (
        all(len(p["indices"]) == 1 for p in paths)
        and len({p["indices"][0] for p in paths}) == 1
    )
    claim = f"{ABBREVIATIONS[direction]}({target},{query.reference})"
    if direct:
        citation = f"P{paths[0]['indices'][0] + 1}"
        return (
            f"[{citation}] {claim}."
            if symbolic
            else f"{citation} places {target} {direction.lower()} of {query.reference}."
        )
    return "\n".join(
        [
            *[_order_text(p, symbolic) for p in paths],
            f"Therefore {claim}."
            if symbolic
            else f"Together, these relations place {target} {direction.lower()} of {query.reference}.",
        ]
    )


def _condition(path, candidate, reference):
    sign = -1 if any(path["signs"]) else 0
    if (path["nodes"][0], path["nodes"][-1]) == (reference, candidate):
        sign = -sign
    elif (path["nodes"][0], path["nodes"][-1]) != (candidate, reference):
        raise VerificationError("membership proof endpoints differ")
    return path["axis"], sign


def _references(paths, chains):
    refs = []
    for path in paths:
        chain = next(
            (i for i, c in enumerate(chains, 1) if contains_path(c, path)), None
        )
        ref = f"Chain {chain}" if chain is not None else _citations(path)
        if ref not in refs:
            refs.append(ref)
    return "; ".join(refs)


def _count_text(problem, value, plan, symbolic):
    lines = []
    for i, path in enumerate(plan["chains"], 1):
        shared = None
        for length in range(len(path["indices"]) - 1, 0, -1):
            prefix = {
                **path,
                "nodes": path["nodes"][: length + 1],
                "indices": path["indices"][:length],
                "signs": path["signs"][:length],
            }
            previous = next(
                (
                    j
                    for j, chain in enumerate(plan["chains"][: i - 1], 1)
                    if contains_path(chain, prefix)
                ),
                None,
            )
            if previous is not None:
                shared = length, previous
                break
        if shared is None:
            text = _order_text(path, symbolic)
        else:
            length, previous = shared
            suffix = {
                **path,
                "nodes": path["nodes"][length:],
                "indices": path["indices"][length:],
                "signs": path["signs"][length:],
            }
            text = (
                f"Reuse Chain {previous} from {path['nodes'][0]} to {path['nodes'][length]}; "
                f"then {_order_text(suffix, symbolic)}"
            )
        lines.append(f"Chain {i}: {text}")
    grouped = defaultdict(list)
    for member in plan["members"]:
        conditions = tuple(
            sorted(
                _condition(p, member["candidate"], problem.query.reference)
                for p in member["paths"]
            )
        )
        grouped[member["qualifies"], conditions].append(member)
    direction = next(iter(problem.query.directions)).value
    reference = problem.query.reference
    for (qualifies, conditions), members in grouped.items():
        names = [
            f"{m['candidate']} ({_references(m['paths'], plan['chains'])})"
            for m in members
        ]
        if symbolic:
            comparisons = " & ".join(
                f"{axis}(o) {'<' if sign < 0 else '>' if sign > 0 else '='} {axis}({reference})"
                for axis, sign in conditions
            )
            lines.append(
                f"For o in {{{'; '.join(names)}}}: {comparisons} => "
                f"{'' if qualifies else 'NOT '}{ABBREVIATIONS[direction]}(o,{reference})."
            )
        elif qualifies:
            lines.append(
                f"{_join(names)} {'is' if len(names) == 1 else 'are'} {direction.lower()} of {reference}."
            )
        else:
            descriptions = []
            for axis, sign in conditions:
                if sign:
                    side = (
                        ("west" if sign < 0 else "east")
                        if axis == "x"
                        else ("south" if sign < 0 else "north")
                    )
                    descriptions.append(f"further {side} than {reference}")
                else:
                    descriptions.append(
                        f"on the same {'vertical' if axis == 'x' else 'horizontal'} line as {reference}"
                    )
            lines.append(
                f"{_join(names)} {'lies' if len(names) == 1 else 'lie'} {_join(descriptions)}, "
                f"so {'it cannot' if len(names) == 1 else 'none can'} be exactly {direction.lower()} of {reference}."
            )
    lines.append(
        f"All scene memberships are fixed; count={value}."
        if symbolic
        else f"Every other scene object has been accounted for. The premises fix the count at {value}."
    )
    return "\n\n".join(lines)


def _conflict_text(conflict, symbolic):
    lines = []
    for edges in conflict.paths:
        path = {
            "axis": "xy"[edges[0].axis],
            "nodes": [edges[0].start, *(e.end for e in edges)],
            "signs": [-1 if e.strict else 0 for e in edges],
            "indices": [e.premise_index for e in edges],
        }
        lines.append(_order_text(path, symbolic))
    if conflict.kind == "strict-cycle":
        edge = conflict.paths[0][0]
        side = "west" if edge.axis == 0 else "south"
        lines.append(
            f"This would put {edge.start} further {side} than itself, which is impossible."
            if not symbolic
            else "A strict ordering cannot return to its starting point."
        )
    else:
        a, b = conflict.paths[0][0].start, conflict.paths[0][-1].end
        lines.append(
            f"{a} and {b} would occupy the same point, but named places must be distinct."
        )
    return "\n".join(lines)


def render(evidence, menu):
    """Render one shared checked evidence plan in both styles."""
    plan = make_plan(evidence)
    audit_plan(evidence, plan)
    semantics, problem = evidence.semantics, evidence.semantics.problem
    decision = encode(semantics, menu, Contract.UNIQUE)
    indices = (
        set(evidence.conflict.premise_indices)
        if not semantics.consistent
        else {i for path in plan["paths"] for i in path["indices"]}
    )
    key = _premise_key(problem, indices)
    pair = {}
    for style in ("natural", "symbolic"):
        symbolic = style == "symbolic"
        lines = [key] if key else []
        if not semantics.consistent:
            lines.extend(
                [
                    _conflict_text(evidence.conflict, symbolic),
                    "The premises contradict one another.",
                ]
            )
        else:
            for value in plan["values"]:
                lines.append(
                    _count_text(problem, value, plan, symbolic)
                    if isinstance(problem.query, CountQuery)
                    else _relation_text(
                        problem, value, evidence.positive_support[value], symbolic
                    )
                )
            for unit in plan["outline"]["units"]:
                text = render_scenario(unit, symbolic, _join, ABBREVIATIONS)
                text = text.replace(
                    "One possibility consistent with all premises is the following.",
                    "One arrangement satisfying the premises is as follows.",
                ).replace(
                    "(possible together under all premises)", "(satisfies the premises)"
                )
                text = text.replace("left to right", "west to east").replace(
                    "bottom to top", "south to north"
                )
                text = text.replace(
                    "horizontal position", "east-west position"
                ).replace("height", "north-south position")
                text = text.replace(
                    "No eligible object qualifies", "No object qualifies"
                )
                lines.append(text)
                if unit["case"] in plan["negative_groups"]:
                    objects = plan["negative_groups"][unit["case"]]
                    direction = next(iter(problem.query.directions)).value.lower()
                    lines.append(
                        f"Case {unit['case']} falsifies the queried relation for {{{', '.join(objects)}}}; none is entailed."
                        if symbolic
                        else f"{_join(objects)} {'is' if len(objects) == 1 else 'are'} not {direction} of {problem.query.reference} "
                        + (
                            "in this arrangement, so it is not guaranteed to be."
                            if len(objects) == 1
                            else "in this arrangement, so none is guaranteed to be."
                        )
                    )
            if plan["exact_ambiguity"]:
                noun = (
                    "direction"
                    if isinstance(problem.query, DirectionQuery)
                    else "exact count"
                )
                lines.append(
                    f"The same premises allow different outcomes, so no {noun} is guaranteed."
                )
            elif len(semantics.entailed) > 1:
                lines.append("There is more than one guaranteed answer.")
            elif isinstance(problem.query, WhichQuery):
                direction = next(iter(problem.query.directions)).value.lower()
                lines.append(
                    f"Only {plan['values'][0]} is guaranteed to be {direction} of {problem.query.reference}."
                    if plan["values"]
                    else f"No object is guaranteed to be {direction} of {problem.query.reference}."
                )
            if len(semantics.entailed) == 1:
                value = plan["values"][0]
                lines.append(
                    f"{value} is option {decision.letters}."
                    if value in menu
                    else f"{value} is not listed among A-D."
                )
        lines.append(f"Answer: {decision.letters}")
        pair[style] = "\n\n".join(lines)
    return pair
