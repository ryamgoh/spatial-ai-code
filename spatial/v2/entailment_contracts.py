"""Entailment-only contract interface; incompatible with possibility-based releases."""

import shlex
from dataclasses import dataclass
from enum import Enum

from spatial.v2.composition_contracts import (
    Decision,
    VerificationError,
    _parse_prompt_fields,
    answer_domain,
    validate_menu,
    witness_outcome,
)
from spatial.v2.composition_contracts import (
    parse_answer as parse_legacy_answer,
)
from spatial.v2.composition_contracts import validate_problem as _validate_problem
from spatial.v2.grading import AnswerMode
from spatial.v2.solver import (
    CountQuery,
    DirectionQuery,
    SpatialProblem,
    SpatialSolverV2,
    WhichQuery,
)

VERSION = "spatialentail-entailment-v1"


def validate_problem(problem):
    _validate_problem(problem, allow_multiword_names=True)


class Contract(str, Enum):
    UNIQUE = "unique-entailed"
    LISTED = "listed-entailed"
    EXHAUSTIVE = "exhaustive-entailed"


@dataclass(frozen=True)
class Semantics:
    problem: SpatialProblem
    consistent: bool
    entailed: frozenset[str | int]
    counterexamples: dict
    engine: str

    @property
    def case(self):
        if not self.consistent:
            return "inconsistent"
        if not self.entailed:
            return "no-entailed-answer"
        return "unique-entailed" if len(self.entailed) == 1 else "multiple-entailed"


def check_counterexample(problem, coordinates, value):
    """Check every premise and the falsity of the claim without solver helpers."""
    outcome = witness_outcome(problem, coordinates)
    holds = (
        value in outcome if isinstance(problem.query, WhichQuery) else value == outcome
    )
    if holds:
        raise VerificationError("counterexample does not falsify the query claim")


def analyze(problem, *, solver=None):
    validate_problem(problem)
    solver = solver or SpatialSolverV2(backend="z3")
    result = solver.analyze_entailment(problem)
    if result.error is not None:
        raise VerificationError(f"inconclusive entailment analysis: {result.error}")
    entailed = frozenset(result.entailed_values)
    domain = set(answer_domain(problem))
    value_type = int if isinstance(problem.query, CountQuery) else str
    if (
        not entailed <= domain
        or len(entailed) != len(result.entailed_values)
        or any(type(v) is not value_type for v in result.entailed_values)
        or any(type(v) is not value_type for v in result.counterexamples)
    ):
        raise VerificationError("invalid entailed-answer set")
    if not result.consistent:
        if entailed or result.counterexamples:
            raise VerificationError("inconsistent premises contain answer evidence")
    else:
        if not isinstance(problem.query, WhichQuery) and len(entailed) > 1:
            raise VerificationError("Direction/Count cannot have two entailed values")
        if set(result.counterexamples) != domain - entailed:
            raise VerificationError(
                "counterexamples must cover every non-entailed claim"
            )
        for value, coordinates in result.counterexamples.items():
            check_counterexample(problem, coordinates, value)
    return Semantics(
        problem, result.consistent, entailed, result.counterexamples, result.engine
    )


def encode(semantics, menu, contract):
    if not isinstance(semantics, Semantics):
        raise TypeError("entailment contracts require entailment-only semantics")
    contract = Contract(contract)
    validate_menu(semantics.problem, menu)
    if not semantics.consistent:
        return Decision("E", semantics.case, "inconsistent")
    selected = "".join(
        letter for letter, value in zip("ABCD", menu) if value in semantics.entailed
    )
    if contract is Contract.UNIQUE:
        if len(semantics.entailed) != 1:
            return Decision("E", semantics.case, semantics.case)
        return Decision(
            selected or "F", semantics.case, None if selected else "unique-unlisted"
        )
    if not semantics.entailed:
        return Decision("F", semantics.case, "no-entailed-answer")
    if contract is Contract.LISTED:
        return Decision(
            selected or "F",
            semantics.case,
            None if selected else "no-listed-entailments",
        )
    if not semantics.entailed <= set(menu):
        return Decision("F", semantics.case, "incomplete-entailed-coverage")
    return Decision(selected, semantics.case)


def parse_answer(text, contract, *, stage=2):
    mode = (
        AnswerMode.SINGLE
        if Contract(contract) is Contract.UNIQUE
        else AnswerMode.VISIBLE_POSSIBLE
    )
    return parse_legacy_answer(text, mode, stage=stage)


ASSUMPTIONS = (
    "Named objects are distinct points on fixed global x/y axes. No two objects "
    "coincide; sharing one coordinate is allowed. Directions are exact coordinate "
    "comparisons: North means equal x and greater y; Northeast means greater x "
    "and greater y. There are no distances, unit steps, grid spacing, or map bounds. "
    "Unstated relations are unknown, not false. A valid arrangement satisfies "
    "every premise and these assumptions. A claim is entailed only when it holds "
    "in every valid arrangement. A claim that is merely compatible with the "
    "premises is not necessarily entailed. Under every contract, select E if "
    "no valid arrangement exists."
)
RULES = {
    Contract.UNIQUE: (
        "Select the one A-D value if exactly one domain answer is entailed. "
        "Select F if that sole entailed answer is unlisted. Select E if zero or "
        "multiple domain answers are entailed. For Selection, count individually "
        "entailed candidates; an additional candidate that is not guaranteed "
        "does not invalidate the sole entailed candidate."
    ),
    Contract.LISTED: (
        "Select all displayed A-D values that are entailed. Exclude every displayed "
        "value that is not entailed, including claims that could hold but are not "
        "guaranteed. Select F if no displayed value is entailed."
    ),
    Contract.EXHAUSTIVE: (
        "Select all A-D values only if they cover the complete nonempty set of "
        "individually entailed domain answers. Select F if any entailed answer is "
        "missing or no domain answer is entailed. Non-entailed answers outside "
        "the menu do not make coverage incomplete."
    ),
}


def _question(problem, contract):
    query = problem.query
    scope = "listed" if contract is Contract.LISTED else "eligible"
    if isinstance(query, DirectionQuery):
        relation = f"{query.target} relative to {query.reference}"
        return {
            Contract.UNIQUE: f"Which single direction of {relation} is entailed?",
            Contract.LISTED: f"Which listed directions of {relation} are entailed? Select all that apply.",
            Contract.EXHAUSTIVE: f"Select the complete set of individually entailed directions of {relation}.",
        }[contract]
    relation = f"{next(iter(query.directions)).value.lower()} of {query.reference}"
    if isinstance(query, WhichQuery):
        return (
            f"Which single eligible candidate is individually guaranteed to be {relation}?"
            if contract is Contract.UNIQUE
            else f"Which {scope} candidates are guaranteed to be {relation}? "
            + (
                "Select all listed entailed candidates."
                if contract is Contract.LISTED
                else "Select the complete entailed-candidate set."
            )
        )
    return (
        f"What exact count of eligible objects {relation} is guaranteed?"
        if contract is Contract.UNIQUE
        else f"Which {'listed ' if contract is Contract.LISTED else ''}exact counts "
        f"of eligible objects {relation} are entailed? "
        + (
            "Select all that apply."
            if contract is Contract.LISTED
            else "Select the complete entailed-count set."
        )
    )


def render_prompt(problem, menu, contract):
    contract = Contract(contract)
    validate_problem(problem)
    validate_menu(problem, menu)
    query = problem.query
    query_record = (
        f"direction {shlex.quote(query.target)} {shlex.quote(query.reference)}"
        if isinstance(query, DirectionQuery)
        else f"{'which' if isinstance(query, WhichQuery) else 'count'} "
        f"{next(iter(query.directions)).value} {shlex.quote(query.reference)}"
    )
    lines = [
        f"Interface: {VERSION}",
        f"Contract: {contract.value}",
        ASSUMPTIONS,
        RULES[contract],
        (
            "Eligible objects are independent of the menu. Count concerns all eligible "
            "objects jointly. Zero is an ordinary exact Count answer."
        ),
        (
            "End with Answer: <letters>. Use ascending uppercase letters without spaces "
            "or punctuation. Unique requires one letter; other contracts allow a "
            "nonempty subset of A-D. E and F must each appear alone."
        ),
        "Objects: " + ", ".join(problem.objects),
        "Query: " + query_record,
    ]
    if not isinstance(query, DirectionQuery):
        lines.append("Eligible: " + ", ".join(query.candidates))
    lines.append("Consider a map with multiple objects:")
    lines.extend(
        f"{atom.subject} is to the {next(iter(atom.allowed)).value} of {atom.reference}."
        for atom in problem.premise.operands
    )
    lines.append("Question: " + _question(problem, contract))
    lines.extend(f"{letter}. {value}" for letter, value in zip("ABCD", menu))
    lines.append("E. Cannot be determined")
    lines.append(
        "F. No complete entailed answer can be formed from A-D."
        if contract is Contract.EXHAUSTIVE
        else "F. None of the above"
    )
    return "\n".join(lines)


def parse_prompt(text):
    problem, menu, contract = _parse_prompt_fields(
        text, Contract, allow_multiword_names=True
    )
    if render_prompt(problem, menu, contract) != text:
        raise ValueError("prompt does not match the entailment-only template")
    return problem, menu, contract
