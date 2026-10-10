"""Natural and symbolic renderings of checked SpatialEntail proof objects."""

from __future__ import annotations

from collections.abc import Mapping

from spatial.v2.proofs import (
    AxisFact,
    Contradiction,
    DirectionClaim,
    DirectionProofCertificate,
    DirectionRefutationCertificate,
    FormulaRefutationCertificate,
    OrderRelation,
    ProofAxis,
    ProofRule,
    check_direction_proof,
    check_direction_refutation,
    check_formula_refutation,
)
from spatial.v2.solver import (
    And,
    Direction,
    Iff,
    Implies,
    Not,
    Or,
    RelationConstraint,
    SpatialFormula,
)
from spatial.v2.symbolic_trace_codec import (
    render_symbolic_direction_proof,
    render_symbolic_direction_refutation,
    render_symbolic_formula_refutation,
)
from spatial.v2.trace import TraceFormat


def _label(value: str, labels: Mapping[str, str]) -> str:
    return labels.get(value, value)


def _direction_atom(
    atom: RelationConstraint,
    labels: Mapping[str, str],
) -> str:
    directions = tuple(
        direction.value for direction in Direction if direction in atom.allowed
    )
    relation = (
        directions[0]
        if len(directions) == 1
        else "one of {" + ", ".join(directions) + "}"
    )
    return (
        f"{_label(atom.subject, labels)} is {relation} of "
        f"{_label(atom.reference, labels)}"
    )


def _formula_text(
    formula: SpatialFormula,
    labels: Mapping[str, str],
    *,
    symbolic: bool,
) -> str:
    if isinstance(formula, RelationConstraint):
        if symbolic:
            directions = tuple(
                direction.name
                for direction in Direction
                if direction in formula.allowed
            )
            relation = (
                directions[0]
                if len(directions) == 1
                else "IN_{" + ",".join(directions) + "}"
            )
            return (
                f"DIR_{relation}({_label(formula.subject, labels)},"
                f"{_label(formula.reference, labels)})"
            )
        return _direction_atom(formula, labels)
    if isinstance(formula, Not):
        operand = _formula_text(formula.operand, labels, symbolic=symbolic)
        return f"NOT ({operand})" if symbolic else f"not ({operand})"
    if isinstance(formula, (And, Or)):
        if symbolic:
            operator = " AND " if isinstance(formula, And) else " OR "
        else:
            operator = " and " if isinstance(formula, And) else " or "
        return operator.join(
            f"({_formula_text(operand, labels, symbolic=symbolic)})"
            for operand in formula.operands
        )
    if isinstance(formula, Implies):
        left = _formula_text(formula.antecedent, labels, symbolic=symbolic)
        right = _formula_text(formula.consequent, labels, symbolic=symbolic)
        return f"({left}) -> ({right})" if symbolic else f"if {left}, then {right}"
    if isinstance(formula, Iff):
        left = _formula_text(formula.left, labels, symbolic=symbolic)
        right = _formula_text(formula.right, labels, symbolic=symbolic)
        return f"({left}) <-> ({right})" if symbolic else f"{left} exactly when {right}"
    raise TypeError(f"unsupported proof formula: {type(formula).__name__}")


def render_formula(
    formula: SpatialFormula,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render a structured formula consistently across proof artifacts."""
    trace_format = TraceFormat(trace_format)
    return _formula_text(
        formula,
        labels or {},
        symbolic=trace_format is TraceFormat.SYMBOLIC,
    )


def _axis_text(fact: AxisFact, labels: Mapping[str, str]) -> str:
    subject = _label(fact.subject, labels)
    reference = _label(fact.reference, labels)
    if fact.relation is OrderRelation.EQUAL:
        axis = "X" if fact.axis is ProofAxis.X else "Y"
        return f"{subject} and {reference} share the same {axis} coordinate"
    if fact.axis is ProofAxis.X:
        relation = "west" if fact.relation is OrderRelation.LESS else "east"
    else:
        relation = "south" if fact.relation is OrderRelation.LESS else "north"
    return f"{subject} is {relation} of {reference}"


def _claim_text(claim: DirectionClaim, labels: Mapping[str, str]) -> str:
    return (
        f"{_label(claim.subject, labels)} is {claim.direction.value} of "
        f"{_label(claim.reference, labels)}"
    )


def _natural_step(step, labels: Mapping[str, str]) -> str:
    conclusion = step.conclusion
    scope = f"[{step.branch}] " if step.branch is not None else ""
    if step.rule is ProofRule.PREMISE:
        assert isinstance(conclusion, SpatialFormula)
        assert step.premise_index is not None
        return f"{step.id}: {_formula_text(conclusion, labels, symbolic=False)}."
    if step.rule is ProofRule.ASSUMPTION:
        assert isinstance(conclusion, SpatialFormula)
        return (
            f"{scope}{step.id}: Assume "
            f"{_formula_text(conclusion, labels, symbolic=False)} from {step.inputs[0]}."
        )
    if step.rule is ProofRule.REFUTATION_ASSUMPTION:
        assert isinstance(conclusion, SpatialFormula)
        return (
            f"{scope}{step.id}: Assume for contradiction that "
            f"{_formula_text(conclusion, labels, symbolic=False)}."
        )
    if step.rule in {
        ProofRule.AND_ELIMINATION,
        ProofRule.AND_INTRODUCTION,
        ProofRule.MODUS_PONENS,
        ProofRule.DISJUNCTIVE_SYLLOGISM,
        ProofRule.IFF_ELIMINATION,
        ProofRule.DOUBLE_NEGATION,
    }:
        assert isinstance(conclusion, SpatialFormula)
        dependencies = " and ".join(step.inputs)
        rule = step.rule.value.replace("-", " ")
        return (
            f"{scope}{step.id}: From {dependencies} by {rule}, "
            f"{_formula_text(conclusion, labels, symbolic=False)}."
        )
    if step.rule is ProofRule.CONTRADICTION:
        assert isinstance(conclusion, Contradiction)
        return f"{scope}{step.id}: {step.inputs[0]} and {step.inputs[1]} contradict."
    if step.rule is ProofRule.EXPLOSION:
        assert isinstance(conclusion, SpatialFormula)
        return (
            f"{scope}{step.id}: Branch {step.branch} is closed by {step.inputs[0]}, "
            f"so {_formula_text(conclusion, labels, symbolic=False)} follows in that case."
        )
    if step.rule is ProofRule.CASE_SPLIT:
        assert isinstance(conclusion, SpatialFormula)
        branches = ", ".join(step.inputs[1:])
        return (
            f"{scope}{step.id}: Every case from {step.inputs[0]} concludes "
            f"{_formula_text(conclusion, labels, symbolic=False)} via {branches}."
        )
    if step.rule is ProofRule.DIRECTION_DECOMPOSITION:
        assert isinstance(conclusion, AxisFact)
        return (
            f"{scope}{step.id}: From {step.inputs[0]} on the "
            f"{conclusion.axis.value.upper()}-axis, {_axis_text(conclusion, labels)}."
        )
    if step.rule is ProofRule.AXIS_INVERSION:
        assert isinstance(conclusion, AxisFact)
        return f"{scope}{step.id}: Equivalently from {step.inputs[0]}, {_axis_text(conclusion, labels)}."
    if step.rule is ProofRule.AXIS_TRANSITIVITY:
        assert isinstance(conclusion, AxisFact)
        return (
            f"{scope}{step.id}: By {conclusion.axis.value.upper()}-axis transitivity from "
            f"{step.inputs[0]} and {step.inputs[1]}, {_axis_text(conclusion, labels)}."
        )
    if step.rule is ProofRule.AXIS_CONTRADICTION:
        assert isinstance(conclusion, Contradiction)
        return (
            f"{scope}{step.id}: {step.inputs[0]} and {step.inputs[1]} assign "
            "different relations to the same axis pair, so the assumption is impossible."
        )
    if step.rule is ProofRule.DIRECTION_INTRODUCTION:
        assert isinstance(conclusion, DirectionClaim)
        return (
            f"{scope}{step.id}: Therefore {_claim_text(conclusion, labels)}, "
            f"as established by {step.inputs[0]}."
        )
    assert step.rule is ProofRule.DIRECTION_RECOMPOSITION
    assert isinstance(conclusion, DirectionClaim)
    return (
        f"{scope}{step.id}: Combining {step.inputs[0]} and {step.inputs[1]} gives "
        f"{_claim_text(conclusion, labels)}."
    )


def render_direction_proof(
    certificate: DirectionProofCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render one checked proof object without consulting solver coordinates."""
    check_direction_proof(certificate)
    trace_format = TraceFormat(trace_format)
    labels = labels or {}
    if trace_format is TraceFormat.SYMBOLIC:
        if labels:
            raise ValueError(
                "symbolic proof traces use certificate identifiers and do not "
                "accept display labels"
            )
        return render_symbolic_direction_proof(certificate)

    return "\n".join(_natural_step(step, labels) for step in certificate.steps)


def render_direction_training_proof(
    certificate: DirectionProofCertificate,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Narrate the same checked derivation dependencies as the symbolic trace."""
    return render_direction_proof(certificate, TraceFormat.NATURAL, labels)


def render_direction_training_refutation(
    certificate: DirectionRefutationCertificate,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Narrate scoped assumptions and every dependency of the contradiction."""
    return render_direction_refutation(certificate, TraceFormat.NATURAL, labels)


def render_direction_refutation(
    certificate: DirectionRefutationCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render one checked contradiction for an impossible candidate."""
    check_direction_refutation(certificate)
    trace_format = TraceFormat(trace_format)
    labels = labels or {}
    if trace_format is TraceFormat.SYMBOLIC:
        if labels:
            raise ValueError(
                "symbolic refutation traces use certificate identifiers and do not "
                "accept display labels"
            )
        return render_symbolic_direction_refutation(certificate)
    lines = [_natural_step(step, labels) for step in certificate.steps]
    claim = render_formula(certificate.claim, trace_format, labels)
    lines.append(f"Therefore the claim that {claim} is impossible.")
    return "\n".join(lines)


def render_formula_refutation(
    certificate: FormulaRefutationCertificate,
    trace_format: TraceFormat | str,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render one checked contradiction for an arbitrary spatial formula."""
    check_formula_refutation(certificate)
    trace_format = TraceFormat(trace_format)
    labels = labels or {}
    if trace_format is TraceFormat.SYMBOLIC:
        if labels:
            raise ValueError(
                "symbolic refutation traces use certificate identifiers and do not "
                "accept display labels"
            )
        return render_symbolic_formula_refutation(certificate)
    lines = [_natural_step(step, labels) for step in certificate.steps]
    claim = render_formula(certificate.claim, trace_format, labels)
    lines.append(f"Therefore the claim that {claim} is impossible.")
    return "\n".join(lines)
