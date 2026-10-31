"""Checked positive paths and negative counterexamples for entailment-only views."""

from dataclasses import dataclass, replace

from spatial.v2.composition_conflicts import (
    build_conflict,
    check_conflict,
    render_conflict,
)
from spatial.v2.composition_contracts import (
    UnsupportedEvidence,
    VerificationError,
    answer_domain,
)
from spatial.v2.composition_depth import membership_support
from spatial.v2.composition_readable import _derive, _model, _path, _render_path
from spatial.v2.composition_scenarios import qualitative_outline, render_scenario
from spatial.v2.entailment_contracts import (
    VERSION,
    Contract,
    Semantics,
    analyze,
    check_counterexample,
    encode,
    parse_answer,
)
from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    WhichQuery,
    direction_signs,
)

ABBREVIATIONS = {
    d.value: label
    for d, label in zip(Direction, ("N", "NE", "E", "SE", "S", "SW", "W", "NW"))
}
REFERENCE_VERSION = "self-contained-premise-keys-v1"
CONCISE_VERSION = "unique-readable-explanations-v2"
SPATIAL_PROSE_VERSION = "spatial-prose-shared-chains-v3"


def reference_options(row):
    version = row.get("reference_version")
    if version == SPATIAL_PROSE_VERSION:
        return {"restate_premises": True, "spatial_prose": True}
    if version == CONCISE_VERSION:
        return {"restate_premises": True, "concise": True}
    if version not in (None, REFERENCE_VERSION):
        raise VerificationError("unknown explanation reference version")
    return {"restate_premises": version == REFERENCE_VERSION}


@dataclass(frozen=True)
class Evidence:
    semantics: Semantics
    positive_support: dict
    conflict: object = None


def _claim_parts(problem, value):
    query = problem.query
    if isinstance(query, DirectionQuery):
        return query.target, query.reference, Direction(value)
    return value, query.reference, next(iter(query.directions))


def _check_paths(problem, paths, subject, reference, direction, *, positive):
    axes = set()
    for path in paths:
        replayed = _path(problem, path["axis"], path["nodes"], path["indices"])
        if replayed != path:
            raise VerificationError("positive proof path failed independent replay")
        axes.add(path["axis"])
        nodes = path["nodes"]
        sign = -1 if any(path["signs"]) else 0
        if (nodes[0], nodes[-1]) == (reference, subject):
            sign = -sign
        elif (nodes[0], nodes[-1]) != (subject, reference):
            raise VerificationError("proof path uses different claim endpoints")
        required = direction_signs(direction)["xy".index(path["axis"])]
        if (sign == required) != positive:
            raise VerificationError("proof axis does not establish its claim")
    if (positive and axes != {"x", "y"}) or not axes:
        raise VerificationError("entailed direction requires both axes")


def _relation_support(problem, value):
    subject, reference, direction = _claim_parts(problem, value)
    paths = _derive(problem, subject, reference, direction)
    _check_paths(problem, paths, subject, reference, direction, positive=True)
    return {"kind": "relation", "paths": paths}


def _count_support(problem, value, solver):
    query = problem.query
    projection = replace(
        problem, query=WhichQuery(query.directions, query.reference, query.candidates)
    )
    memberships = analyze(projection, solver=solver)
    supports = []
    direction = next(iter(query.directions))
    for candidate in query.candidates:
        qualifies = candidate in memberships.entailed
        record = membership_support(projection, candidate, qualifies)
        paths = [
            _path(
                projection,
                p["axis"],
                p["path"],
                p["premise_indices"],
            )
            for p in record["paths"]
        ]
        _check_paths(
            projection, paths, candidate, query.reference, direction, positive=qualifies
        )
        supports.append(
            {"candidate": candidate, "qualifies": qualifies, "paths": paths}
        )
    if sum(item["qualifies"] for item in supports) != value:
        raise VerificationError("fixed memberships contradict entailed count")
    return {"kind": "fixed-count", "memberships": supports}


def build_evidence(problem, *, solver=None):
    semantics = analyze(problem, solver=solver)
    conflict, supports = None, {}
    if not semantics.consistent:
        conflict = build_conflict(problem, allow_multiword_names=True)
        check_conflict(conflict, allow_multiword_names=True)
    else:
        for value in answer_domain(problem):
            if value not in semantics.entailed:
                continue
            supports[value] = (
                _count_support(problem, value, solver)
                if isinstance(problem.query, CountQuery)
                else _relation_support(problem, value)
            )
    evidence = Evidence(semantics, supports, conflict)
    verify_evidence(evidence, solver=solver)
    return evidence


def verify_evidence(evidence, *, solver=None):
    semantics = evidence.semantics
    actual = analyze(semantics.problem, solver=solver)
    if (actual.consistent, actual.entailed) != (
        semantics.consistent,
        semantics.entailed,
    ):
        raise VerificationError("evidence disagrees with direct entailment checks")
    if set(evidence.positive_support) != semantics.entailed:
        raise VerificationError("positive proofs must cover the entailed-answer set")
    domain = set(answer_domain(semantics.problem))
    expected_negatives = domain - semantics.entailed if semantics.consistent else set()
    if set(semantics.counterexamples) != expected_negatives:
        raise VerificationError("negative evidence does not cover the domain")
    for value, coordinates in semantics.counterexamples.items():
        check_counterexample(semantics.problem, coordinates, value)
    if not semantics.consistent:
        if evidence.conflict is None or evidence.conflict.problem != semantics.problem:
            raise VerificationError("inconsistency requires matching conflict evidence")
        check_conflict(evidence.conflict, allow_multiword_names=True)
        subset = replace(
            semantics.problem,
            premise=And(
                tuple(
                    semantics.problem.premise.operands[i]
                    for i in evidence.conflict.premise_indices
                )
            ),
        )
        if analyze(subset, solver=solver).consistent:
            raise VerificationError("cited conflict subset is consistent")
        return
    if evidence.conflict is not None:
        raise VerificationError("consistent evidence contains a conflict")
    for value, support in evidence.positive_support.items():
        if support["kind"] == "relation":
            if isinstance(semantics.problem.query, CountQuery):
                raise VerificationError("Count evidence requires joint count support")
            parts = _claim_parts(semantics.problem, value)
            _check_paths(semantics.problem, support["paths"], *parts, positive=True)
        elif support["kind"] == "fixed-count":
            query = semantics.problem.query
            if not isinstance(query, CountQuery):
                raise VerificationError("fixed-count support used for another query")
            memberships = support["memberships"]
            if [item["candidate"] for item in memberships] != list(query.candidates):
                raise VerificationError(
                    "Count proof must cover every eligible candidate"
                )
            for item in memberships:
                if type(item["qualifies"]) is not bool:
                    raise VerificationError("invalid fixed membership status")
                _check_paths(
                    semantics.problem,
                    item["paths"],
                    item["candidate"],
                    query.reference,
                    next(iter(query.directions)),
                    positive=item["qualifies"],
                )
            if sum(item["qualifies"] for item in memberships) != value:
                raise VerificationError("fixed Count total is incorrect")
        else:
            raise UnsupportedEvidence("unknown entailment proof support")


def _join(values):
    values = list(map(str, values))
    if not values:
        return "none"
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + " and " + values[-1]


def _claim_text(problem, value, symbolic):
    if isinstance(problem.query, CountQuery):
        return (
            f"count={value}"
            if symbolic
            else "exactly 1 eligible candidate qualifies"
            if value == 1
            else f"exactly {value} eligible candidates qualify"
        )
    subject, reference, direction = _claim_parts(problem, value)
    return (
        f"{ABBREVIATIONS[direction.value]}({subject},{reference})"
        if symbolic
        else f"{subject} is {direction.value.lower()} of {reference}"
    )


def _source_path(path, symbolic, restate_premises):
    if restate_premises:
        path = {**path, "indices": [f"P{i + 1}" for i in path["indices"]]}
    return _render_path(path, symbolic)


def _premise_key(problem, indices):
    if not indices:
        return ""
    lines = ["Premises used (labels follow their order in the question):"]
    for i in sorted(indices):
        atom = problem.premise.operands[i]
        direction = next(iter(atom.allowed)).value
        lines.append(
            f"P{i + 1}: {atom.subject} is to the {direction} of {atom.reference}."
        )
    return "\n".join(lines)


def _positive_text(problem, value, support, symbolic, restate_premises=False):
    lines = []
    if support["kind"] == "fixed-count":
        for item in support["memberships"]:
            lines.extend(
                _source_path(p, symbolic, restate_premises) for p in item["paths"]
            )
            candidate, qualifies = item["candidate"], item["qualifies"]
            direction = next(iter(problem.query.directions)).value
            lines.append(
                f"{'' if qualifies else 'NOT '}{ABBREVIATIONS[direction]}({candidate},{problem.query.reference}) in every valid arrangement."
                if symbolic
                else f"{candidate} {'always qualifies' if qualifies else 'never qualifies'} for the queried relation."
            )
        names = [
            item["candidate"] for item in support["memberships"] if item["qualifies"]
        ]
        lines.append(
            f"Fixed qualifying set={{{', '.join(names)}}}; count={value} in every valid arrangement."
            if symbolic
            else f"The qualifying candidates are fixed: {_join(names)}. Exactly {value} "
            f"{'qualifies' if value == 1 else 'qualify'} in every valid arrangement."
        )
    else:
        lines.extend(
            _source_path(p, symbolic, restate_premises) for p in support["paths"]
        )
        subject, reference, direction = _claim_parts(problem, value)
        lines.append(
            f"{ABBREVIATIONS[direction.value]}({subject},{reference}) in every valid arrangement."
            if symbolic
            else f"Therefore, {subject} is {direction.value.lower()} of {reference} in every valid arrangement."
        )
    return "\n".join(lines)


def render_pair(
    evidence,
    menu,
    contract,
    *,
    solver=None,
    restate_premises=False,
    concise=False,
    spatial_prose=False,
):
    verify_evidence(evidence, solver=solver)
    contract = Contract(contract)
    if spatial_prose:
        if contract is not Contract.UNIQUE or not restate_premises or concise:
            raise ValueError(
                "spatial prose requires self-contained Unique references and its own version"
            )
        from spatial.v2.entailment_readable_v3 import render

        return render(evidence, menu)
    if concise and (contract is not Contract.UNIQUE or not restate_premises):
        raise ValueError(
            "concise explanations require self-contained Unique references"
        )
    semantics, problem = evidence.semantics, evidence.semantics.problem
    decision = encode(semantics, menu, contract)
    # Evidence is selected by obligations, never differently for the two styles.
    domain = answer_domain(problem)
    selected = menu if contract is Contract.LISTED else domain
    if decision.escape_reason == "incomplete-entailed-coverage" or (
        decision.escape_reason == "unique-unlisted"
        and not isinstance(problem.query, WhichQuery)
    ):
        selected = tuple(
            v for v in domain if v in semantics.entailed and v not in menu
        )[:1]
    elif contract is Contract.UNIQUE and len(semantics.entailed) > 1:
        selected = tuple(v for v in domain if v in semantics.entailed)[:2]
    if (
        concise
        and len(semantics.entailed) == 1
        and not isinstance(problem.query, WhichQuery)
    ):
        selected = tuple(v for v in domain if v in semantics.entailed)
    negatives = [v for v in selected if v not in semantics.entailed]
    outline = qualitative_outline(
        [_model(problem, semantics.counterexamples[v]) for v in negatives]
        if semantics.consistent
        else [],
        problem,
        VERSION,
    )
    indices = set()
    if restate_premises:
        if not semantics.consistent:
            indices.update(evidence.conflict.premise_indices)
        else:
            for value in selected:
                if value not in semantics.entailed:
                    continue
                support = evidence.positive_support[value]
                paths = (
                    support["paths"]
                    if support["kind"] == "relation"
                    else [
                        path
                        for member in support["memberships"]
                        for path in member["paths"]
                    ]
                )
                indices.update(i for path in paths for i in path["indices"])
    key = _premise_key(problem, indices)
    pair = {}
    for style in ("natural", "symbolic"):
        symbolic = style == "symbolic"
        lines = [key] if key else []
        if not semantics.consistent:
            conflict_text = render_conflict(
                evidence.conflict,
                symbolic=symbolic,
                allow_multiword_names=True,
                premise_prefix="P" if restate_premises else "p",
            )
            if concise:
                conflict_text = conflict_text.replace(
                    " No minimality claim is made.", ""
                )
            lines.append(conflict_text)
        else:
            for value in selected:
                if value in semantics.entailed:
                    lines.append(
                        _positive_text(
                            problem,
                            value,
                            evidence.positive_support[value],
                            symbolic,
                            restate_premises,
                        )
                    )
            for unit in outline["units"]:
                text = render_scenario(unit, symbolic, _join, ABBREVIATIONS)
                text = text.replace(
                    "One possibility consistent with all premises is the following.",
                    "A counterexample arrangement satisfying all premises is the following.",
                ).replace(
                    "(possible together under all premises)",
                    "(counterexample satisfying all premises)",
                )
                lines.append(text)
            for value, witness in zip(negatives, outline["audit_witnesses"]):
                claim = _claim_text(problem, value, symbolic)
                lines.append(
                    f"The claim that {claim} is false in Case {witness['case']}, "
                    "so it is not entailed."
                )
        reasons = {
            "inconsistent": "The premises are inconsistent; every contract requires E.",
            "no-entailed-answer": (
                "No displayed answer is entailed."
                if contract is Contract.LISTED
                else "No domain answer is entailed."
            ),
            "multiple-entailed": "At least two domain answers are entailed; Unique requires exactly one.",
            "unique-unlisted": "The sole entailed answer is unlisted.",
            "no-listed-entailments": "No displayed answer is entailed.",
            "incomplete-entailed-coverage": "An entailed answer is missing from A-D, so the complete entailed set cannot be represented.",
        }
        if concise:
            reasons.update(
                {
                    "inconsistent": "The premises contradict one another, so the answer is E.",
                    "no-entailed-answer": "No single answer value is guaranteed by the premises, so the answer is E.",
                    "multiple-entailed": "More than one object is guaranteed to satisfy the question, so there is no single answer. Choose E.",
                }
            )
            if len(semantics.entailed) == 1:
                value = next(iter(semantics.entailed))
                if not isinstance(problem.query, WhichQuery):
                    noun = (
                        "count"
                        if isinstance(problem.query, CountQuery)
                        else "direction"
                    )
                    lines.append(
                        f"Exact {noun} values are mutually exclusive; this is the sole entailed {noun}."
                    )
                reasons["unique-unlisted"] = (
                    f"The sole entailed answer is {value}, but it is not among A-D. Choose F."
                )
        if decision.escape_reason:
            lines.append(reasons[decision.escape_reason])
        else:
            values = [v for v in menu if v in semantics.entailed]
            lines.append(
                f"The sole entailed answer is {values[0]}, listed as {decision.letters}."
                if concise
                else f"The {'displayed' if contract is Contract.LISTED else 'complete'} entailed-answer set is {{{', '.join(map(str, values))}}}."
            )
        lines.append(f"Answer: {decision.letters}")
        pair[style] = "\n\n".join(lines)
        if parse_answer(pair[style], contract, stage=1) != decision.letters:
            raise VerificationError("explanation answer disagrees with semantics")
    return pair


def check_pair(
    evidence,
    menu,
    contract,
    pair,
    *,
    solver=None,
    restate_premises=False,
    concise=False,
    spatial_prose=False,
):
    """Validate template output against checked evidence, not arbitrary model prose."""
    if pair != render_pair(
        evidence,
        menu,
        contract,
        solver=solver,
        restate_premises=restate_premises,
        concise=concise,
        spatial_prose=spatial_prose,
    ):
        raise VerificationError("references differ from the checked matched rendering")
