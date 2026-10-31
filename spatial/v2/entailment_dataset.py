"""Entailment-only admission and audits; no sealing or automatic training."""

import random
from dataclasses import replace

from spatial.v2.composition_contracts import (
    UnsupportedEvidence,
    VerificationError,
    answer_domain,
)
from spatial.v2.composition_depth import _record, membership_support
from spatial.v2.composition_feasibility import (
    graph_profile,
    stable_int,
    token_measurement,
)
from spatial.v2.composition_pilot import renamed_problem
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.difficulty import derive_axis_proof
from spatial.v2.entailment_contracts import (
    VERSION,
    Contract,
    analyze,
    encode,
    render_prompt,
)
from spatial.v2.entailment_evidence import (
    REFERENCE_VERSION,
    build_evidence,
    check_pair,
    reference_options,
    render_pair,
)
from spatial.v2.entailment_scene_protocol import (
    PLAIN_VERSIONS,
    SHORT_VERSION,
    display_references,
    parse_row,
    require_scene_scope,
)
from spatial.v2.entailment_scene_protocol import (
    SCOPE as SCENE_SCOPE,
)
from spatial.v2.entailment_scene_protocol import (
    render_prompt as render_scene_prompt,
)
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import And, Direction, DirectionQuery, WhichQuery


def signature(semantics):
    return semantics.consistent, semantics.entailed


def _support_paths(value):
    if isinstance(value, dict):
        if value.get("kind") == "path":
            yield value
        else:
            for part in value.values():
                yield from _support_paths(part)
    elif isinstance(value, (tuple, list)):
        for part in value:
            yield from _support_paths(part)


def uniqueness_failure_depth(evidence):
    """Depth of the cheapest pair of positive membership certificates."""
    if (
        not isinstance(evidence.semantics.problem.query, WhichQuery)
        or len(evidence.semantics.entailed) < 2
    ):
        return None
    depths = sorted(
        max(len(path["indices"]) for path in _support_paths(support))
        for support in evidence.positive_support.values()
    )
    return depths[1]


def selected_explanation_depth(evidence, menu, contract):
    """Measure the positive certificate actually emitted by the Unique renderer."""
    if Contract(contract) is not Contract.UNIQUE or not evidence.semantics.consistent:
        return None
    semantics = evidence.semantics
    decision = encode(semantics, menu, contract)
    values = [
        value
        for value in answer_domain(semantics.problem)
        if value in semantics.entailed
    ]
    if len(values) == 1:
        return max(
            len(path["indices"])
            for path in _support_paths(evidence.positive_support[values[0]])
        )
    if decision.escape_reason == "multiple-entailed":
        # The renderer emits the first two domain members, not an arbitrary
        # deepest pair. Credit only a cheapest-pair certificate.
        emitted = max(
            len(path["indices"])
            for value in values[:2]
            for path in _support_paths(evidence.positive_support[value])
        )
        if emitted == uniqueness_failure_depth(evidence):
            return emitted
    return None


def verify_transformations(story, problem, semantics, solver, *, scene_scope=False):
    variant, names = renamed_problem(problem)
    expected = (
        semantics.consistent,
        frozenset(names[v] for v in semantics.entailed)
        if isinstance(problem.query, WhichQuery)
        else semantics.entailed,
    )
    if signature(analyze(variant, solver=solver)) != expected:
        raise VerificationError("rename/order changed entailment semantics")
    for removed in [*((i,) for i in story.distractors), story.distractors]:
        if not removed:
            continue
        retained = tuple(
            atom for i, atom in enumerate(problem.premise.operands) if i not in removed
        )
        if not retained:
            raise UnsupportedEvidence("distractor removal leaves no atomic premises")
        reduced = replace(problem, premise=And(retained))
        if signature(analyze(reduced, solver=solver)) != signature(semantics):
            if scene_scope:
                raise UnsupportedEvidence(
                    "previously scoped distractor changes all-scene answers"
                )
            raise VerificationError("distractor removal changed entailment answers")


def measure_depth(problem, *, solver=None):
    semantics = analyze(problem, solver=solver)
    if not semantics.consistent:
        raise UnsupportedEvidence("inconsistent: no entailment-depth descriptor")
    query = problem.query
    if isinstance(query, DirectionQuery):
        if not semantics.entailed:
            raise UnsupportedEvidence("no entailed direction")
        proof = derive_axis_proof(
            problem,
            query.target,
            query.reference,
            Direction(next(iter(semantics.entailed))),
        )
        if proof is None:
            raise UnsupportedEvidence("entailed direction lacks axis support")
        return {"version": "entailment-axis-depth-v1", **_record((proof.x, proof.y))}
    projection = replace(
        problem, query=WhichQuery(query.directions, query.reference, query.candidates)
    )
    memberships = analyze(projection, solver=solver)
    supports = [
        membership_support(projection, candidate, candidate in memberships.entailed)
        for candidate in query.candidates
    ]
    depth = max(item["depth"] for item in supports)
    for item in supports:
        if item["depth"] != depth:
            continue
        for index in item["premise_indices"]:
            retained = tuple(
                atom for i, atom in enumerate(problem.premise.operands) if i != index
            )
            if retained and signature(
                analyze(replace(problem, premise=And(retained)), solver=solver)
            ) != signature(semantics):
                return {
                    "version": "entailment-axis-depth-v1",
                    "depth": depth,
                    "memberships": supports,
                    "dependence_check": "entailed-answer-changed",
                    "removed_premise_index": index,
                }
    raise UnsupportedEvidence("deepest support does not change the entailed-answer set")


def menus_for(semantics, seed, *, omission_percent=20):
    if type(omission_percent) is not int or not 0 <= omission_percent <= 100:
        raise ValueError("omission percent must be an integer in 0-100")
    domain = answer_domain(semantics.problem)
    if len(domain) < 4:
        raise UnsupportedEvidence("four real menu values are required")
    entailed = [v for v in domain if v in semantics.entailed]
    other = [v for v in domain if v not in semantics.entailed]
    rng = random.Random(seed)
    rng.shuffle(entailed)
    rng.shuffle(other)
    menu = (entailed + other)[:4]
    rng.shuffle(menu)
    replacements = [v for v in other if v not in menu]
    if (
        semantics.consistent
        and entailed
        and semantics.entailed <= set(menu)
        and replacements
        and seed % 100 < omission_percent
    ):
        partial = menu.copy()
        index = rng.choice([i for i, v in enumerate(menu) if v in semantics.entailed])
        partial[index] = rng.choice(replacements)
        return tuple(menu), tuple(partial)
    return (tuple(menu),)


def audit_row(row, evidence, *, solver=None):
    if row.get("interface") != VERSION:
        raise VerificationError("row is not entailment-only")
    problem, menu, contract = parse_row(row)
    if (
        problem != evidence.semantics.problem
        or list(menu) != row["menu"]
        or contract.value != row["contract"]
    ):
        raise VerificationError("row problem/menu/contract differs from its evidence")
    if encode(evidence.semantics, menu, contract).letters != row["answer"]:
        raise VerificationError("row gold differs from entailment computation")
    entailed_letters = "".join(
        letter
        for letter, value in zip("ABCD", menu)
        if value in evidence.semantics.entailed
    )
    if row["entailed_menu_letters"] != entailed_letters:
        raise VerificationError("row entailed-option metadata is incorrect")
    categories = (
        solver.negative_categories(problem)
        if solver is not None
        else QuestionSolver().negative_categories(problem)
    )
    expected_categories = {
        letter: categories[value]
        for letter, value in zip("ABCD", menu)
        if value in categories
    }
    if row["negative_menu_categories"] != expected_categories:
        raise VerificationError("row negative-category metadata is incorrect")
    for category in ("contingent", "impossible"):
        if row["has_" + category + "_negative"] != (
            category in expected_categories.values()
        ):
            raise VerificationError("row negative coverage flag is incorrect")
    if row["sole_entailed_with_contingent"] != (
        len(evidence.semantics.entailed) == 1 and "contingent" in categories.values()
    ):
        raise VerificationError("row sole-entailment coverage flag is incorrect")
    failure_depth = (
        uniqueness_failure_depth(evidence) if contract is Contract.UNIQUE else None
    )
    if "uniqueness_failure_depth" in row and (
        failure_depth is None or row["uniqueness_failure_depth"] != failure_depth
    ):
        raise VerificationError("row uniqueness-failure depth is incorrect")
    expected_support = row.get("substantive_depth_support", False)
    if failure_depth is not None:
        expected_support = row["depth"] is not None and failure_depth == row["depth"]
    if (
        "decision_depth_support" in row
        and row["decision_depth_support"] is not expected_support
    ):
        raise VerificationError("row decision-depth support is incorrect")
    if "selected_explanation_depth" in row:
        selected_depth = selected_explanation_depth(evidence, menu, contract)
        if row["selected_explanation_depth"] != selected_depth or row.get(
            "selected_explanation_depth_support"
        ) is not (selected_depth is not None and selected_depth == row["depth"]):
            raise VerificationError("selected explanation depth is incorrect")
    if row.get("display_names") is None:
        check_pair(
            evidence,
            menu,
            contract,
            row["references"],
            solver=solver,
            **reference_options(row),
        )
    elif row["references"] != display_references(
        render_pair(evidence, menu, contract, solver=solver, **reference_options(row)),
        row["display_names"],
    ):
        raise VerificationError("displayed references differ from checked evidence")


def build_attempt(
    story,
    problem,
    requested_depth,
    tokenizer,
    *,
    solver=None,
    omission_percent=20,
    prompt_version=None,
    display_names=None,
):
    if prompt_version not in (None, *PLAIN_VERSIONS):
        raise ValueError("unknown spatial prompt protocol")
    plain = prompt_version in PLAIN_VERSIONS
    if plain:
        require_scene_scope(problem)
    elif display_names is not None:
        raise ValueError("display-name mapping requires the plain scene protocol")
    solver = solver or QuestionSolver()
    kind = type(problem.query).__name__.removesuffix("Query").lower()
    base_id = f"{story.story_id}-{kind}"
    record = {
        "interface": VERSION,
        "base_id": base_id,
        "story_id": story.story_id,
        "family_id": story.source_family,
        "recipe": story.recipe,
        "query_kind": kind,
        "requested_depth": requested_depth,
        "problem": tagged_dataclass_to_dict(problem),
        "graph": graph_profile(story),
        "stage": "entailment-analysis",
    }
    if prompt_version is not None:
        record.update(prompt_version=prompt_version, scope_policy=SCENE_SCOPE)
    if prompt_version == SHORT_VERSION:
        from spatial.v2.entailment_unique_study import STUDY_SCOPE

        record["study_scope"] = STUDY_SCOPE
    try:
        evidence = build_evidence(problem, solver=solver)
        semantics = evidence.semantics
        categories = solver.negative_categories(problem)
        record["semantics"] = {
            "consistent": semantics.consistent,
            "entailed": sorted(semantics.entailed),
            "case": semantics.case,
            "domain_size": len(answer_domain(problem)),
            "negative_categories": {
                str(value): category for value, category in categories.items()
            },
        }
        record["stage"] = "metamorphic-checks"
        verify_transformations(
            story,
            problem,
            semantics,
            solver,
            scene_scope=plain,
        )
        record["stage"] = "depth"
        try:
            measured = measure_depth(problem, solver=solver)
        except UnsupportedEvidence as exc:
            measured = {"depth": None, "reason": str(exc)}
        if requested_depth is not None and measured["depth"] != requested_depth:
            raise UnsupportedEvidence(
                f"depth mismatch: {requested_depth} vs {measured['depth']}"
            )
        record["measurement"] = measured
        failure_depth = uniqueness_failure_depth(evidence)
        rows, tokens = [], []
        record["stage"] = "render-and-audit"
        menus = menus_for(
            semantics, stable_int(base_id), omission_percent=omission_percent
        )
        for index, menu in enumerate(menus):
            for contract in (
                (Contract.UNIQUE,) if prompt_version == SHORT_VERSION else Contract
            ):
                decision = encode(semantics, menu, contract)
                row = {
                    "interface": VERSION,
                    "id": f"{base_id}-m{index}-{contract.value}",
                    "base_id": base_id,
                    "story_id": story.story_id,
                    "family_id": story.source_family,
                    "bundle_id": f"{base_id}-m{index}",
                    "contract": contract.value,
                    "query_kind": kind,
                    "split": "development-verification-only",
                    "menu": list(menu),
                    "menu_role": "primary" if index == 0 else "omission",
                    "depth": measured["depth"],
                    "semantic_case": semantics.case,
                    "has_contingent_negative": any(
                        categories.get(v) == "contingent" for v in menu
                    ),
                    "has_impossible_negative": any(
                        categories.get(v) == "impossible" for v in menu
                    ),
                    "sole_entailed_with_contingent": (
                        len(semantics.entailed) == 1
                        and "contingent" in categories.values()
                    ),
                    "entailed_count": len(semantics.entailed),
                    "displayed_entailed_count": len(set(menu) & semantics.entailed),
                    "entailed_menu_letters": "".join(
                        letter
                        for letter, v in zip("ABCD", menu)
                        if v in semantics.entailed
                    ),
                    "negative_menu_categories": {
                        letter: categories[v]
                        for letter, v in zip("ABCD", menu)
                        if v in categories
                    },
                    "answer": decision.letters,
                    "escape_reason": decision.escape_reason,
                    "coverage": "full"
                    if semantics.entailed <= set(menu)
                    else "partial",
                    "prompt": (
                        render_scene_prompt(
                            problem,
                            menu,
                            contract,
                            display_names=display_names,
                            prompt_version=prompt_version,
                        )
                        if plain
                        else render_prompt(problem, menu, contract)
                    ),
                    "references": display_references(
                        render_pair(
                            evidence,
                            menu,
                            contract,
                            solver=solver,
                            restate_premises=plain,
                        ),
                        display_names,
                    ),
                }
                if prompt_version is not None:
                    row.update(
                        prompt_version=prompt_version,
                        scope_policy=SCENE_SCOPE,
                        reference_version=REFERENCE_VERSION,
                    )
                if display_names is not None:
                    row["display_names"] = dict(display_names)
                if prompt_version == SHORT_VERSION:
                    row["study_scope"] = record["study_scope"]
                visible_support = (
                    set(menu) & semantics.entailed
                    if contract is Contract.LISTED
                    else semantics.entailed
                )
                row["substantive_depth_support"] = (
                    decision.letters not in ("E", "F")
                    and measured["depth"] is not None
                    and any(
                        len(path["indices"]) == measured["depth"]
                        for value in visible_support
                        for path in _support_paths(evidence.positive_support[value])
                    )
                )
                row["decision_depth_support"] = row["substantive_depth_support"]
                if contract is Contract.UNIQUE and failure_depth is not None:
                    row["uniqueness_failure_depth"] = failure_depth
                    row["decision_depth_support"] = (
                        decision.escape_reason == "multiple-entailed"
                        and measured["depth"] is not None
                        and failure_depth == measured["depth"]
                    )
                if prompt_version == SHORT_VERSION:
                    selected_depth = selected_explanation_depth(
                        evidence, menu, contract
                    )
                    row["selected_explanation_depth"] = selected_depth
                    row["selected_explanation_depth_support"] = (
                        selected_depth is not None
                        and selected_depth == measured["depth"]
                    )
                audit_row(row, evidence, solver=solver)
                reversed_menu = tuple(reversed(menu))
                before = {
                    menu["ABCD".index(c)] for c in decision.letters if c in "ABCD"
                }
                after_decision = encode(semantics, reversed_menu, contract)
                after = {
                    reversed_menu["ABCD".index(c)]
                    for c in after_decision.letters
                    if c in "ABCD"
                }
                if before != after or (
                    decision.letters in ("E", "F")
                    and decision.letters != after_decision.letters
                ):
                    raise VerificationError(
                        "menu permutation changed entailment selection"
                    )
                rows.append(row)
                tokens.extend(
                    token_measurement(tokenizer, row, style)
                    for style in row["references"]
                )
        record.update(
            stage="complete",
            rows=rows,
            tokens=tokens,
            evidence=tagged_dataclass_to_dict(evidence),
            status="accepted"
            if all(t["admitted"] for t in tokens)
            else "token-rejected",
        )
    except (ValueError, RuntimeError) as exc:
        record.update(
            status="unsupported"
            if isinstance(exc, UnsupportedEvidence)
            else "verification-failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
    return record
