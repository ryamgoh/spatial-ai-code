"""Strict JSON grammars for replayable symbolic training traces."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from spatial.v2.answer_certificates import (
    DirectionAnswerSetCertificate,
    DirectionEntailmentCertificate,
    check_direction_answer_set,
)
from spatial.v2.count_certificates import (
    CountAnswerSetCertificate,
    CountAssignmentRefutation,
    CountImpossibilityCertificate,
    CountValueCertificate,
    check_count_answer_set,
    exact_count_formula,
)
from spatial.v2.grading import (
    MenuAnswer,
    SymbolicAnswerDecision,
    check_symbolic_answer_decision,
    parse_symbolic_answer_decision,
)
from spatial.v2.model_certificates import (
    ContingencyCertificate,
    CoordinateAssignment,
    SpatialModelCertificate,
    check_contingency_certificate,
    check_model_certificate,
)
from spatial.v2.proofs import (
    AxisFact,
    Contradiction,
    DirectionClaim,
    DirectionProofCertificate,
    DirectionRefutationCertificate,
    FormulaRefutationCertificate,
    OrderRelation,
    ProofAxis,
    ProofConclusion,
    ProofRule,
    ProofStep,
    check_direction_proof,
    check_direction_refutation,
    check_formula_refutation,
)
from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    Iff,
    Implies,
    Not,
    Or,
    RelationConstraint,
    SpatialFormula,
    SpatialProblem,
    WhichQuery,
    direction_constraint,
    membership_constraint,
)
from spatial.v2.which_certificates import (
    MembershipEntailmentCertificate,
    MembershipImpossibilityCertificate,
    MembershipProofCertificate,
    WhichAnswerSetCertificate,
    WhichCandidateCertificate,
    check_which_answer_set,
)

DIRECTION_PROOF_SCHEMA = "spatial-direction-proof-v1"
DIRECTION_REFUTATION_SCHEMA = "spatial-direction-refutation-v1"
FORMULA_REFUTATION_SCHEMA = "spatial-formula-refutation-v1"


class SymbolicProofError(ValueError):
    """A symbolic proof does not conform to the replayable JSON grammar."""


@dataclass(frozen=True)
class CheckedDirectionTrace:
    """Replayed evidence extracted from one symbolic Direction trace."""

    problem: SpatialProblem
    possible_directions: tuple[Direction, ...]
    proof: DirectionProofCertificate | None
    models: tuple[SpatialModelCertificate, ...]
    refutations: tuple[DirectionRefutationCertificate, ...]


@dataclass(frozen=True)
class CheckedSymbolicTrainingTrace:
    """A replayed reasoning envelope paired with its checked menu decision."""

    reasoning: (
        CheckedDirectionTrace | WhichAnswerSetCertificate | CountAnswerSetCertificate
    )
    decision: SymbolicAnswerDecision


@dataclass(frozen=True)
class SymbolicTraceScore:
    """Process-level validity for one model-produced Symbolic trace."""

    reasoning_valid: bool
    decision_valid: bool
    domain_consistent: bool
    fully_valid: bool
    error_stage: str | None = None
    error: str | None = None


def _formula_payload(formula: SpatialFormula) -> dict[str, Any]:
    if isinstance(formula, RelationConstraint):
        return {
            "kind": "relation",
            "subject": formula.subject,
            "reference": formula.reference,
            "directions": [
                direction.name
                for direction in Direction
                if direction in formula.allowed
            ],
        }
    if isinstance(formula, Not):
        return {"kind": "not", "operand": _formula_payload(formula.operand)}
    if isinstance(formula, (And, Or)):
        return {
            "kind": "and" if isinstance(formula, And) else "or",
            "operands": [_formula_payload(item) for item in formula.operands],
        }
    if isinstance(formula, Implies):
        return {
            "kind": "implies",
            "antecedent": _formula_payload(formula.antecedent),
            "consequent": _formula_payload(formula.consequent),
        }
    if isinstance(formula, Iff):
        return {
            "kind": "iff",
            "left": _formula_payload(formula.left),
            "right": _formula_payload(formula.right),
        }
    raise TypeError(f"unsupported spatial formula: {type(formula).__name__}")


def _conclusion_payload(conclusion: ProofConclusion) -> dict[str, Any]:
    if isinstance(conclusion, SpatialFormula):
        return _formula_payload(conclusion)
    if isinstance(conclusion, AxisFact):
        return {
            "kind": "axis",
            "axis": conclusion.axis.value,
            "subject": conclusion.subject,
            "relation": conclusion.relation.value,
            "reference": conclusion.reference,
        }
    if isinstance(conclusion, DirectionClaim):
        return {
            "kind": "direction",
            "subject": conclusion.subject,
            "direction": conclusion.direction.name,
            "reference": conclusion.reference,
        }
    if isinstance(conclusion, Contradiction):
        return {"kind": "contradiction"}
    raise TypeError(f"unsupported proof conclusion: {type(conclusion).__name__}")


def render_symbolic_direction_proof(
    certificate: DirectionProofCertificate,
) -> str:
    """Serialize one checked proof in the canonical compact JSON grammar."""
    check_direction_proof(certificate)
    payload = {
        "schema": DIRECTION_PROOF_SCHEMA,
        "conclusion_step": certificate.conclusion_step,
        "steps": _steps_payload(certificate.steps),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_symbolic_direction_refutation(
    certificate: DirectionRefutationCertificate,
) -> str:
    """Serialize one checked refutation in the canonical compact JSON grammar."""
    check_direction_refutation(certificate)
    return _render_refutation(certificate, DIRECTION_REFUTATION_SCHEMA)


def render_symbolic_formula_refutation(
    certificate: FormulaRefutationCertificate,
) -> str:
    """Serialize one checked formula refutation in the compact JSON grammar."""
    check_formula_refutation(certificate)
    return _render_refutation(certificate, FORMULA_REFUTATION_SCHEMA)


def _render_refutation(
    certificate: DirectionRefutationCertificate | FormulaRefutationCertificate,
    schema: str,
) -> str:
    payload = {
        "schema": schema,
        "claim": _formula_payload(certificate.claim),
        "assumption_step": certificate.assumption_step,
        "contradiction_step": certificate.contradiction_step,
        "steps": _steps_payload(certificate.steps),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _steps_payload(steps: tuple[ProofStep, ...]) -> list[dict[str, Any]]:
    return [
        {
            "id": step.id,
            "rule": step.rule.value,
            "conclusion": _conclusion_payload(step.conclusion),
            "inputs": list(step.inputs),
            "premise_index": step.premise_index,
            "branch": step.branch,
        }
        for step in steps
    ]


def _axis_groups(
    certificate: SpatialModelCertificate,
    axis: str,
) -> list[list[str]]:
    coordinate = (lambda item: item.x) if axis == "x" else (lambda item: item.y)
    groups: dict[int, list[str]] = {}
    for assignment in certificate.assignments:
        groups.setdefault(coordinate(assignment), []).append(assignment.object)
    return [sorted(groups[value]) for value in sorted(groups)]


def _model_payload(certificate: SpatialModelCertificate) -> dict[str, Any]:
    check_model_certificate(certificate)
    return {
        "kind": "model",
        "claim": _formula_payload(certificate.claim),
        "expected": certificate.expected_claim_value,
        "x_order": _axis_groups(certificate, "x"),
        "y_order": _axis_groups(certificate, "y"),
    }


def render_symbolic_direction_trace(
    certificate: DirectionAnswerSetCertificate,
) -> str:
    """Serialize compact, replayable evidence for one Direction answer set."""
    check_direction_answer_set(certificate)
    evidence = []
    if certificate.is_unique:
        item = next(item for item in certificate.candidates if item.entailed)
        assert isinstance(item.evidence, DirectionEntailmentCertificate)
        evidence.append(
            {
                "evidence": json.loads(
                    render_symbolic_direction_proof(item.evidence.proof)
                ),
                "direction": item.direction.name,
                "status": "entailed",
                "witness": _model_payload(item.evidence.witness),
            }
        )
        mode = "unique"
    else:
        for item in certificate.candidates:
            if isinstance(item.evidence, SpatialModelCertificate):
                value = _model_payload(item.evidence)
                status = "possible"
            else:
                value = json.loads(render_symbolic_direction_refutation(item.evidence))
                status = "impossible"
            evidence.append(
                {
                    "evidence": value,
                    "direction": item.direction.name,
                    "status": status,
                }
            )
        mode = "ambiguous"
    payload = {
        "schema": "spatial-direction-trace-v2",
        "mode": mode,
        "evidence": evidence,
        "possible_directions": [
            direction.name for direction in certificate.possible_directions
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _membership_evidence_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, MembershipProofCertificate):
        return {
            "kind": "membership-proof",
            "witness": _model_payload(value.witness),
            "proof": json.loads(render_symbolic_direction_proof(value.proof)),
        }
    if isinstance(value, MembershipEntailmentCertificate):
        return {
            "kind": "membership-entailment",
            "witness": _model_payload(value.witness),
            "refutations": [
                json.loads(render_symbolic_direction_refutation(item))
                for item in value.refutations
            ],
        }
    if isinstance(value, MembershipImpossibilityCertificate):
        return {
            "kind": "membership-impossibility",
            "countermodel": _model_payload(value.countermodel),
            "refutations": [
                json.loads(render_symbolic_direction_refutation(item))
                for item in value.refutations
            ],
        }
    if isinstance(value, ContingencyCertificate):
        check_contingency_certificate(value)
        return {
            "kind": "contingency",
            "witness": _model_payload(value.witness),
            "counterexample": _model_payload(value.counterexample),
        }
    raise TypeError(f"unsupported membership evidence: {type(value).__name__}")


def render_symbolic_which_trace(certificate: WhichAnswerSetCertificate) -> str:
    """Serialize a complete replayable Which training trace."""
    check_which_answer_set(certificate)
    payload = {
        "schema": "spatial-which-trace-v1",
        "evidence": [
            {
                "evidence": _membership_evidence_payload(item.evidence),
                "candidate": item.candidate,
                "status": item.status.value,
            }
            for item in certificate.candidates
        ],
        "possible_entities": list(certificate.possible_entities),
        "entailed_entities": list(certificate.entailed_entities),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _count_assignment_payload(
    assignment: CountAssignmentRefutation,
) -> dict[str, Any]:
    return {
        "members": list(assignment.members),
        "refutation": json.loads(
            render_symbolic_formula_refutation(assignment.refutation)
        ),
    }


def render_symbolic_count_trace(certificate: CountAnswerSetCertificate) -> str:
    """Serialize a complete replayable Count training trace."""
    check_count_answer_set(certificate)
    evidence = []
    for item in certificate.values:
        if isinstance(item.evidence, SpatialModelCertificate):
            value = _model_payload(item.evidence)
            del value["claim"]
            status = "possible"
        else:
            value = {
                "kind": "count-impossibility",
                "assignments": [
                    _count_assignment_payload(assignment)
                    for assignment in item.evidence.assignments
                ],
            }
            status = "impossible"
        evidence.append(
            {
                "evidence": value,
                "count": item.count,
                "status": status,
            }
        )
    payload = {
        "schema": "spatial-count-trace-v2",
        "fixed_memberships": [
            {
                "candidate": item.candidate,
                "evidence": _membership_evidence_payload(item.evidence),
            }
            for item in certificate.fixed_memberships
        ],
        "evidence": evidence,
        "possible_counts": list(certificate.possible_counts),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SymbolicProofError(f"{context} must be a JSON object")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise SymbolicProofError(f"{context} has unexpected or missing fields")


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise SymbolicProofError(f"{context} must be a non-empty string")
    return value


def _formula(value: Any) -> SpatialFormula:
    payload = _object(value, "formula")
    kind = payload.get("kind")
    try:
        if kind == "relation":
            _keys(
                payload,
                {"kind", "subject", "reference", "directions"},
                "relation",
            )
            raw_directions = payload["directions"]
            if not isinstance(raw_directions, list) or not raw_directions:
                raise SymbolicProofError("relation directions must be a non-empty list")
            directions = frozenset(
                Direction[_string(item, "direction")] for item in raw_directions
            )
            return RelationConstraint(
                _string(payload["subject"], "relation subject"),
                _string(payload["reference"], "relation reference"),
                directions,
            )
        if kind == "not":
            _keys(payload, {"kind", "operand"}, "negation")
            return Not(_formula(payload["operand"]))
        if kind in {"and", "or"}:
            _keys(payload, {"kind", "operands"}, str(kind))
            raw_operands = payload["operands"]
            if not isinstance(raw_operands, list):
                raise SymbolicProofError(f"{kind} operands must be a list")
            operands = tuple(_formula(item) for item in raw_operands)
            return And(operands) if kind == "and" else Or(operands)
        if kind == "implies":
            _keys(payload, {"kind", "antecedent", "consequent"}, "implication")
            return Implies(
                _formula(payload["antecedent"]),
                _formula(payload["consequent"]),
            )
        if kind == "iff":
            _keys(payload, {"kind", "left", "right"}, "equivalence")
            return Iff(_formula(payload["left"]), _formula(payload["right"]))
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SymbolicProofError):
            raise
        raise SymbolicProofError(f"invalid {kind} formula") from exc
    raise SymbolicProofError(f"unknown formula kind: {kind!r}")


def _conclusion(value: Any) -> ProofConclusion:
    payload = _object(value, "conclusion")
    kind = payload.get("kind")
    if kind in {"relation", "not", "and", "or", "implies", "iff"}:
        return _formula(payload)
    try:
        if kind == "axis":
            _keys(
                payload,
                {"kind", "axis", "subject", "relation", "reference"},
                "axis fact",
            )
            return AxisFact(
                ProofAxis(_string(payload["axis"], "axis")),
                _string(payload["subject"], "axis subject"),
                OrderRelation(_string(payload["relation"], "axis relation")),
                _string(payload["reference"], "axis reference"),
            )
        if kind == "direction":
            _keys(
                payload,
                {"kind", "subject", "direction", "reference"},
                "direction claim",
            )
            return DirectionClaim(
                _string(payload["subject"], "direction subject"),
                Direction[_string(payload["direction"], "direction")],
                _string(payload["reference"], "direction reference"),
            )
        if kind == "contradiction":
            _keys(payload, {"kind"}, "contradiction")
            return Contradiction()
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SymbolicProofError):
            raise
        raise SymbolicProofError(f"invalid {kind} conclusion") from exc
    raise SymbolicProofError(f"unknown conclusion kind: {kind!r}")


def _step(value: Any) -> ProofStep:
    payload = _object(value, "step")
    _keys(
        payload,
        {"id", "rule", "conclusion", "inputs", "premise_index", "branch"},
        "step",
    )
    raw_inputs = payload["inputs"]
    if not isinstance(raw_inputs, list):
        raise SymbolicProofError("step inputs must be a list")
    inputs = tuple(_string(item, "step input") for item in raw_inputs)
    premise_index = payload["premise_index"]
    if premise_index is not None and type(premise_index) is not int:
        raise SymbolicProofError("premise_index must be an integer or null")
    branch = payload["branch"]
    if branch is not None:
        branch = _string(branch, "branch")
    try:
        rule = ProofRule(_string(payload["rule"], "proof rule"))
    except ValueError as exc:
        raise SymbolicProofError("unknown proof rule") from exc
    return ProofStep(
        _string(payload["id"], "step id"),
        rule,
        _conclusion(payload["conclusion"]),
        inputs,
        premise_index,
        branch,
    )


def parse_symbolic_direction_proof(
    problem: SpatialProblem,
    text: str,
) -> DirectionProofCertificate:
    """Parse and replay one proof emitted in the canonical JSON grammar."""
    try:
        payload = _object(json.loads(text), "proof document")
    except json.JSONDecodeError as exc:
        raise SymbolicProofError("symbolic proof must be valid JSON") from exc
    _keys(payload, {"schema", "conclusion_step", "steps"}, "proof document")
    if payload["schema"] != DIRECTION_PROOF_SCHEMA:
        raise SymbolicProofError(
            f"unsupported symbolic proof schema: {payload['schema']!r}"
        )
    raw_steps = payload["steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise SymbolicProofError("proof steps must be a non-empty list")
    certificate = DirectionProofCertificate(
        problem,
        tuple(_step(item) for item in raw_steps),
        _string(payload["conclusion_step"], "conclusion_step"),
    )
    check_direction_proof(certificate)
    return certificate


def parse_symbolic_direction_refutation(
    problem: SpatialProblem,
    text: str,
) -> DirectionRefutationCertificate:
    """Parse and replay one refutation emitted in the canonical JSON grammar."""
    claim, steps, assumption_step, contradiction_step = _parse_refutation(
        text,
        DIRECTION_REFUTATION_SCHEMA,
        "direction refutation",
    )
    if not isinstance(claim, RelationConstraint):
        raise SymbolicProofError("direction refutation claim must be a relation")
    certificate = DirectionRefutationCertificate(
        problem,
        claim,
        steps,
        assumption_step,
        contradiction_step,
    )
    check_direction_refutation(certificate)
    return certificate


def parse_symbolic_formula_refutation(
    problem: SpatialProblem,
    text: str,
) -> FormulaRefutationCertificate:
    """Parse and replay one formula refutation in the canonical JSON grammar."""
    claim, steps, assumption_step, contradiction_step = _parse_refutation(
        text,
        FORMULA_REFUTATION_SCHEMA,
        "formula refutation",
    )
    certificate = FormulaRefutationCertificate(
        problem,
        claim,
        steps,
        assumption_step,
        contradiction_step,
    )
    check_formula_refutation(certificate)
    return certificate


def _parse_refutation(
    text: str,
    schema: str,
    context: str,
) -> tuple[SpatialFormula, tuple[ProofStep, ...], str, str]:
    try:
        payload = _object(json.loads(text), f"{context} document")
    except json.JSONDecodeError as exc:
        raise SymbolicProofError(f"symbolic {context} must be valid JSON") from exc
    _keys(
        payload,
        {
            "schema",
            "claim",
            "assumption_step",
            "contradiction_step",
            "steps",
        },
        f"{context} document",
    )
    if payload["schema"] != schema:
        raise SymbolicProofError(
            f"unsupported symbolic {context} schema: {payload['schema']!r}"
        )
    raw_steps = payload["steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise SymbolicProofError("refutation steps must be a non-empty list")
    return (
        _formula(payload["claim"]),
        tuple(_step(item) for item in raw_steps),
        _string(payload["assumption_step"], "assumption_step"),
        _string(payload["contradiction_step"], "contradiction_step"),
    )


def _order_coordinates(
    problem: SpatialProblem,
    x_order: Any,
    y_order: Any,
) -> tuple[CoordinateAssignment, ...]:
    def ranks(value: Any, axis: str) -> dict[str, int]:
        if not isinstance(value, list) or not value:
            raise SymbolicProofError(f"{axis}_order must be a non-empty list")
        result: dict[str, int] = {}
        for rank, raw_group in enumerate(value):
            if not isinstance(raw_group, list) or not raw_group:
                raise SymbolicProofError(
                    f"every {axis}_order rank must be a non-empty list"
                )
            for raw_name in raw_group:
                name = _string(raw_name, f"{axis}_order object")
                if name in result:
                    raise SymbolicProofError(
                        f"{axis}_order contains duplicate object {name!r}"
                    )
                result[name] = rank
        if set(result) != set(problem.objects):
            raise SymbolicProofError(
                f"{axis}_order must cover exactly the problem objects"
            )
        return result

    x_ranks = ranks(x_order, "x")
    y_ranks = ranks(y_order, "y")
    return tuple(
        CoordinateAssignment(name, x_ranks[name], y_ranks[name])
        for name in problem.objects
    )


def _model(
    problem: SpatialProblem,
    value: Any,
) -> SpatialModelCertificate:
    payload = _object(value, "model evidence")
    _keys(
        payload,
        {"kind", "claim", "expected", "x_order", "y_order"},
        "model evidence",
    )
    if payload["kind"] != "model" or type(payload["expected"]) is not bool:
        raise SymbolicProofError("invalid model evidence kind or expected value")
    certificate = SpatialModelCertificate(
        problem,
        _formula(payload["claim"]),
        _order_coordinates(problem, payload["x_order"], payload["y_order"]),
        payload["expected"],
    )
    check_model_certificate(certificate)
    return certificate


def _directions(value: Any, context: str) -> tuple[Direction, ...]:
    if not isinstance(value, list) or not value:
        raise SymbolicProofError(f"{context} must be a non-empty list")
    try:
        directions = tuple(Direction[_string(item, context)] for item in value)
    except KeyError as exc:
        raise SymbolicProofError(f"{context} contains an unknown direction") from exc
    if len(directions) != len(set(directions)):
        raise SymbolicProofError(f"{context} contains duplicate directions")
    return directions


def parse_symbolic_direction_trace(
    problem: SpatialProblem,
    text: str,
) -> CheckedDirectionTrace:
    """Parse and replay a compact symbolic Direction answer trace."""
    query = problem.query
    if not isinstance(query, DirectionQuery):
        raise SymbolicProofError("Direction traces require a DirectionQuery")
    try:
        payload = _object(json.loads(text), "Direction trace")
    except json.JSONDecodeError as exc:
        raise SymbolicProofError("symbolic Direction trace must be valid JSON") from exc
    _keys(
        payload,
        {"schema", "mode", "evidence", "possible_directions"},
        "Direction trace",
    )
    if payload["schema"] != "spatial-direction-trace-v2":
        raise SymbolicProofError(
            f"unsupported symbolic Direction trace schema: {payload['schema']!r}"
        )
    possible = _directions(payload["possible_directions"], "possible_directions")
    if any(direction not in query.candidate_directions for direction in possible):
        raise SymbolicProofError("possible directions fall outside the query domain")
    raw_evidence = payload["evidence"]
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise SymbolicProofError("Direction trace evidence must be a non-empty list")

    proof = None
    models = []
    refutations = []
    seen = []
    for raw_item in raw_evidence:
        item = _object(raw_item, "Direction evidence")
        fields = {"evidence", "direction", "status"}
        if item.get("status") == "entailed":
            fields.add("witness")
        _keys(item, fields, "Direction evidence")
        try:
            direction = Direction[_string(item["direction"], "evidence direction")]
        except KeyError as exc:
            raise SymbolicProofError(
                "Direction evidence has an unknown direction"
            ) from exc
        if direction in seen or direction not in query.candidate_directions:
            raise SymbolicProofError("Direction evidence has an invalid candidate")
        seen.append(direction)
        status = item["status"]
        evidence = _object(item["evidence"], "Direction candidate evidence")
        claim = direction_constraint(query.target, query.reference, direction)
        if status == "entailed":
            if proof is not None or evidence.get("schema") != DIRECTION_PROOF_SCHEMA:
                raise SymbolicProofError("invalid entailed Direction evidence")
            proof = parse_symbolic_direction_proof(
                problem,
                json.dumps(evidence, ensure_ascii=False, separators=(",", ":")),
            )
            if proof.conclusion.direction is not direction:
                raise SymbolicProofError("proof concludes the wrong direction")
            witness = _model(problem, item["witness"])
            if witness.claim != claim or witness.expected_claim_value is not True:
                raise SymbolicProofError(
                    "consistency witness certifies the wrong direction"
                )
            models.append(witness)
        elif status == "possible":
            model = _model(problem, evidence)
            if model.claim != claim or model.expected_claim_value is not True:
                raise SymbolicProofError("model certifies the wrong direction")
            models.append(model)
        elif status == "impossible":
            refutation = parse_symbolic_direction_refutation(
                problem,
                json.dumps(evidence, ensure_ascii=False, separators=(",", ":")),
            )
            if refutation.claim != claim:
                raise SymbolicProofError("refutation certifies the wrong direction")
            refutations.append(refutation)
        else:
            raise SymbolicProofError(f"unknown Direction status: {status!r}")

    mode = payload["mode"]
    if mode == "unique":
        if len(possible) != 1 or proof is None or tuple(seen) != possible:
            raise SymbolicProofError("invalid unique Direction evidence")
    elif mode == "ambiguous":
        expected_order = tuple(
            direction
            for direction in Direction
            if direction in query.candidate_directions
        )
        if len(possible) < 2 or proof is not None or tuple(seen) != expected_order:
            raise SymbolicProofError("invalid ambiguous Direction evidence")
        model_directions = tuple(
            direction
            for direction, raw_item in zip(seen, raw_evidence, strict=True)
            if raw_item["status"] == "possible"
        )
        if model_directions != possible:
            raise SymbolicProofError(
                "possible Direction evidence does not match domain"
            )
    else:
        raise SymbolicProofError(f"unknown Direction trace mode: {mode!r}")

    return CheckedDirectionTrace(
        problem,
        possible,
        proof,
        tuple(models),
        tuple(refutations),
    )


def _string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SymbolicProofError(f"{context} must be a list")
    result = tuple(_string(item, context) for item in value)
    if len(result) != len(set(result)):
        raise SymbolicProofError(f"{context} contains duplicates")
    return result


def _direction_projection(
    problem: SpatialProblem,
    query: WhichQuery,
    candidate: str,
) -> SpatialProblem:
    return SpatialProblem(
        problem.objects,
        problem.premise,
        DirectionQuery(candidate, query.reference),
    )


def _direction_refutations(
    problem: SpatialProblem,
    value: Any,
) -> tuple[DirectionRefutationCertificate, ...]:
    if not isinstance(value, list):
        raise SymbolicProofError("membership refutations must be a list")
    return tuple(
        parse_symbolic_direction_refutation(
            problem,
            json.dumps(item, ensure_ascii=False, separators=(",", ":")),
        )
        for item in value
    )


def _membership_evidence(
    problem: SpatialProblem,
    query: WhichQuery,
    candidate: str,
    value: Any,
) -> Any:
    payload = _object(value, "membership evidence")
    kind = payload.get("kind")
    projection = _direction_projection(problem, query, candidate)
    if kind == "membership-proof":
        _keys(payload, {"kind", "witness", "proof"}, "membership proof")
        return MembershipProofCertificate(
            _model(problem, payload["witness"]),
            parse_symbolic_direction_proof(
                projection,
                json.dumps(payload["proof"], ensure_ascii=False, separators=(",", ":")),
            ),
        )
    if kind == "membership-entailment":
        _keys(
            payload,
            {"kind", "witness", "refutations"},
            "membership entailment",
        )
        return MembershipEntailmentCertificate(
            _model(problem, payload["witness"]),
            _direction_refutations(projection, payload["refutations"]),
        )
    if kind == "membership-impossibility":
        _keys(
            payload,
            {"kind", "countermodel", "refutations"},
            "membership impossibility",
        )
        return MembershipImpossibilityCertificate(
            _model(problem, payload["countermodel"]),
            _direction_refutations(projection, payload["refutations"]),
        )
    if kind == "contingency":
        _keys(
            payload,
            {"kind", "witness", "counterexample"},
            "membership contingency",
        )
        claim = membership_constraint(candidate, query)
        certificate = ContingencyCertificate(
            problem,
            claim,
            _model(problem, payload["witness"]),
            _model(problem, payload["counterexample"]),
        )
        check_contingency_certificate(certificate)
        return certificate
    raise SymbolicProofError(f"unknown membership evidence kind: {kind!r}")


def parse_symbolic_which_trace(
    problem: SpatialProblem,
    text: str,
) -> WhichAnswerSetCertificate:
    """Parse and replay a complete symbolic Which answer trace."""
    query = problem.query
    if not isinstance(query, WhichQuery):
        raise SymbolicProofError("Which traces require a WhichQuery")
    try:
        payload = _object(json.loads(text), "Which trace")
    except json.JSONDecodeError as exc:
        raise SymbolicProofError("symbolic Which trace must be valid JSON") from exc
    _keys(
        payload,
        {"schema", "evidence", "possible_entities", "entailed_entities"},
        "Which trace",
    )
    if payload["schema"] != "spatial-which-trace-v1":
        raise SymbolicProofError(
            f"unsupported symbolic Which trace schema: {payload['schema']!r}"
        )
    raw_evidence = payload["evidence"]
    if not isinstance(raw_evidence, list):
        raise SymbolicProofError("Which trace evidence must be a list")
    candidates = []
    for raw_item in raw_evidence:
        item = _object(raw_item, "Which candidate evidence")
        _keys(item, {"evidence", "candidate", "status"}, "Which candidate evidence")
        candidate = _string(item["candidate"], "Which candidate")
        evidence = _membership_evidence(
            problem,
            query,
            candidate,
            item["evidence"],
        )
        candidate_certificate = WhichCandidateCertificate(candidate, evidence)
        if item["status"] != candidate_certificate.status.value:
            raise SymbolicProofError(
                "Which candidate status disagrees with its evidence"
            )
        candidates.append(candidate_certificate)
    certificate = WhichAnswerSetCertificate(problem, tuple(candidates))
    check_which_answer_set(certificate)
    possible = _string_tuple(payload["possible_entities"], "possible_entities")
    entailed = _string_tuple(payload["entailed_entities"], "entailed_entities")
    if (
        possible != certificate.possible_entities
        or entailed != certificate.entailed_entities
    ):
        raise SymbolicProofError("Which domains disagree with candidate evidence")
    return certificate


def _count_assignment(
    problem: SpatialProblem,
    query: CountQuery,
    value: Any,
) -> CountAssignmentRefutation:
    payload = _object(value, "count assignment")
    _keys(payload, {"members", "refutation"}, "count assignment")
    members = _string_tuple(payload["members"], "count assignment members")
    raw_refutation = _object(payload["refutation"], "count assignment refutation")
    refutation = parse_symbolic_formula_refutation(
        problem, json.dumps(raw_refutation, ensure_ascii=False, separators=(",", ":"))
    )
    return CountAssignmentRefutation(members, refutation)


def _count(value: Any, context: str) -> int:
    if type(value) is not int or value < 0:
        raise SymbolicProofError(f"{context} must be a non-negative integer")
    return value


def parse_symbolic_count_trace(
    problem: SpatialProblem,
    text: str,
) -> CountAnswerSetCertificate:
    """Parse and replay a complete symbolic Count answer trace."""
    query = problem.query
    if not isinstance(query, CountQuery):
        raise SymbolicProofError("Count traces require a CountQuery")
    try:
        payload = _object(json.loads(text), "Count trace")
    except json.JSONDecodeError as exc:
        raise SymbolicProofError("symbolic Count trace must be valid JSON") from exc
    _keys(
        payload,
        {"schema", "fixed_memberships", "evidence", "possible_counts"},
        "Count trace",
    )
    if payload["schema"] != "spatial-count-trace-v2":
        raise SymbolicProofError(
            f"unsupported symbolic Count trace schema: {payload['schema']!r}"
        )
    raw_fixed = payload["fixed_memberships"]
    if not isinstance(raw_fixed, list):
        raise SymbolicProofError("fixed_memberships must be a list")
    which_query = WhichQuery(query.directions, query.reference, query.candidates)
    which_problem = SpatialProblem(problem.objects, problem.premise, which_query)
    fixed = []
    for raw_item in raw_fixed:
        item = _object(raw_item, "fixed membership")
        _keys(item, {"candidate", "evidence"}, "fixed membership")
        candidate = _string(item["candidate"], "fixed candidate")
        fixed.append(
            WhichCandidateCertificate(
                candidate,
                _membership_evidence(
                    which_problem, which_query, candidate, item["evidence"]
                ),
            )
        )
    raw_values = payload["evidence"]
    if not isinstance(raw_values, list):
        raise SymbolicProofError("Count trace evidence must be a list")
    values = []
    for raw_item in raw_values:
        item = _object(raw_item, "Count value evidence")
        _keys(item, {"evidence", "count", "status"}, "Count value evidence")
        count = _count(item["count"], "candidate count")
        raw_evidence = _object(item["evidence"], "Count evidence")
        if item["status"] == "possible":
            _keys(
                raw_evidence, {"kind", "expected", "x_order", "y_order"}, "count model"
            )
            evidence: SpatialModelCertificate | CountImpossibilityCertificate = _model(
                problem,
                {
                    **raw_evidence,
                    "claim": _formula_payload(exact_count_formula(query, count)),
                },
            )
        elif item["status"] == "impossible":
            _keys(
                raw_evidence,
                {"kind", "assignments"},
                "count impossibility evidence",
            )
            if raw_evidence["kind"] != "count-impossibility" or not isinstance(
                raw_evidence["assignments"],
                list,
            ):
                raise SymbolicProofError("invalid count impossibility evidence")
            evidence = CountImpossibilityCertificate(
                tuple(
                    _count_assignment(problem, query, assignment)
                    for assignment in raw_evidence["assignments"]
                )
            )
        else:
            raise SymbolicProofError(f"unknown Count status: {item['status']!r}")
        values.append(CountValueCertificate(count, evidence))

    certificate = CountAnswerSetCertificate(problem, tuple(values), tuple(fixed))
    check_count_answer_set(certificate)
    raw_possible = payload["possible_counts"]
    if not isinstance(raw_possible, list):
        raise SymbolicProofError("possible_counts must be a list")
    possible = tuple(_count(item, "possible count") for item in raw_possible)
    if len(possible) != len(set(possible)):
        raise SymbolicProofError("possible_counts contains duplicates")
    if possible != certificate.possible_counts:
        raise SymbolicProofError("Count domain disagrees with candidate evidence")
    return certificate


def parse_symbolic_training_trace(
    problem: SpatialProblem,
    text: str,
    expected: MenuAnswer,
) -> CheckedSymbolicTrainingTrace:
    """Parse and replay a complete two-record symbolic training trace."""
    records = text.splitlines()
    if len(records) != 2 or any(not record for record in records):
        raise SymbolicProofError(
            "symbolic training trace must contain reasoning and decision records"
        )
    reasoning = _parse_symbolic_reasoning(problem, records[0])
    possible = _reasoning_domain(reasoning)
    decision = parse_symbolic_answer_decision(records[1])
    check_symbolic_answer_decision(decision, expected)
    if possible != decision.possible_values:
        raise SymbolicProofError(
            "reasoning evidence disagrees with the answer-decision domain"
        )
    return CheckedSymbolicTrainingTrace(reasoning, decision)


def _parse_symbolic_reasoning(
    problem: SpatialProblem,
    text: str,
) -> CheckedDirectionTrace | WhichAnswerSetCertificate | CountAnswerSetCertificate:
    if isinstance(problem.query, DirectionQuery):
        return parse_symbolic_direction_trace(problem, text)
    if isinstance(problem.query, WhichQuery):
        return parse_symbolic_which_trace(problem, text)
    if isinstance(problem.query, CountQuery):
        return parse_symbolic_count_trace(problem, text)
    raise SymbolicProofError("unsupported query type in symbolic training trace")


def _reasoning_domain(
    reasoning: (
        CheckedDirectionTrace | WhichAnswerSetCertificate | CountAnswerSetCertificate
    ),
) -> tuple[Any, ...]:
    if isinstance(reasoning, CheckedDirectionTrace):
        return reasoning.possible_directions
    if isinstance(reasoning, WhichAnswerSetCertificate):
        return reasoning.possible_entities
    return reasoning.possible_counts


def score_symbolic_training_trace(
    problem: SpatialProblem,
    text: str,
    expected: MenuAnswer,
) -> SymbolicTraceScore:
    """Score syntax, replay, decision, and cross-record consistency separately."""
    records = text.splitlines()
    if len(records) != 2 or any(not record for record in records):
        return SymbolicTraceScore(
            False,
            False,
            False,
            False,
            "envelope",
            "symbolic trace must contain exactly two non-empty records",
        )

    reasoning = None
    reasoning_error = None
    try:
        reasoning = _parse_symbolic_reasoning(problem, records[0])
    except ValueError as exc:
        reasoning_error = str(exc)

    decision = None
    decision_error = None
    try:
        decision = parse_symbolic_answer_decision(records[1])
        check_symbolic_answer_decision(decision, expected)
    except ValueError as exc:
        decision = None
        decision_error = str(exc)

    domain_consistent = bool(
        reasoning is not None
        and decision is not None
        and _reasoning_domain(reasoning) == decision.possible_values
    )
    fully_valid = reasoning is not None and decision is not None and domain_consistent
    if reasoning_error is not None:
        stage, error = "reasoning", reasoning_error
    elif decision_error is not None:
        stage, error = "decision", decision_error
    elif not domain_consistent:
        stage, error = "domain", "reasoning and decision domains disagree"
    else:
        stage, error = None, None
    return SymbolicTraceScore(
        reasoning is not None,
        decision is not None,
        domain_consistent,
        fully_valid,
        stage,
        error,
    )
