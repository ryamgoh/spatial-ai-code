"""Plain scene prompts with explicit, versioned all-nonreference query scope."""

import re
from dataclasses import replace

from spatial.v2.composition_contracts import validate_menu
from spatial.v2.entailment_contracts import ASSUMPTIONS, Contract, validate_problem
from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    SpatialProblem,
    WhichQuery,
    direction_constraint,
)

VERSION = "plain-all-scene-entailment-v1"
SHORT_VERSION = "plain-unique-system-rules-v1"
PLAIN_VERSIONS = (VERSION, SHORT_VERSION)
SCOPE = "all-nonreference-scene-objects-v1"
INTRO = "Consider a map with multiple objects:"
COMMON_INSTRUCTION = (
    ASSUMPTIONS
    + " For Selection and Count, consider every object named in the premises except "
    "the reference object, independently of which options are displayed. Zero is "
    "an ordinary exact Count answer. Follow the answer contract in the question. "
    "End with Answer: <letters>, using ascending uppercase letters without spaces "
    "or punctuation. E and F must each appear alone."
)
UNIQUE_INSTRUCTION = (
    ASSUMPTIONS
    + " For Which and Count, consider every object named in the premises except "
    "the reference object, independently of which options are displayed. "
    "Direction ranges over all eight exact directions. Count asks for an exact "
    "total, including zero; it is not the number of individually entailed members. "
    "Select its A-D option when exactly one answer value is entailed. Select F "
    "when that sole entailed value is not listed. Select E when no answer value "
    "is entailed, multiple answer values are entailed, or the premises are "
    "inconsistent. For Which, exactly one means one individually entailed object; "
    "another object that could qualify but is not guaranteed does not invalidate "
    "that sole entailment. When multiple objects are entailed, E indicates that "
    "there is no single answer, not uncertainty about those objects. "
    "End with exactly one line 'Answer: X', where X is one uppercase A-F letter."
)
SHORT_QUESTIONS = {
    ("direction", Contract.UNIQUE): (
        "In which direction is {target} relative to {reference}?"
    ),
    ("which", Contract.UNIQUE): "Which object is {direction} of {reference}?",
    ("count", Contract.UNIQUE): "How many objects are {direction} of {reference}?",
}
QUESTIONS = {
    ("direction", Contract.UNIQUE): (
        "What single direction of {target} relative to {reference} is guaranteed? "
        "Select E if no direction is guaranteed, or F if it is not listed."
    ),
    ("direction", Contract.LISTED): (
        "Which listed directions of {target} relative to {reference} are guaranteed? "
        "Select all that apply, or F if none are guaranteed."
    ),
    ("direction", Contract.EXHAUSTIVE): (
        "Which directions of {target} relative to {reference} are guaranteed? "
        "Select the complete set, or F if any guaranteed direction is missing "
        "from the options or there are none."
    ),
    ("which", Contract.UNIQUE): (
        "Which single object is guaranteed to be {direction} of {reference}? "
        "Select E if there is not exactly one such object in the scene, "
        "or F if that object is not listed."
    ),
    ("which", Contract.LISTED): (
        "Which listed objects are guaranteed to be {direction} of {reference}? "
        "Select all that apply, or F if none are guaranteed."
    ),
    ("which", Contract.EXHAUSTIVE): (
        "Which objects are guaranteed to be {direction} of {reference}? "
        "Select the complete set, or F if any guaranteed object is missing "
        "from the options or there are none."
    ),
    ("count", Contract.UNIQUE): (
        "What exact number of objects {direction} of {reference} is guaranteed? "
        "Select E if no exact count is guaranteed, or F if that count is not listed."
    ),
    ("count", Contract.LISTED): (
        "Which listed exact counts of objects {direction} of {reference} are "
        "guaranteed? Select all that apply, or F if none are guaranteed."
    ),
    ("count", Contract.EXHAUSTIVE): (
        "Which exact counts of objects {direction} of {reference} are guaranteed? "
        "Select the complete set, or F if any guaranteed count is missing "
        "from the options or there are none."
    ),
}


def all_scene_problem(problem):
    """Change the query domain, never reuse labels from the previous domain."""
    validate_problem(problem)
    named = {
        name
        for atom in problem.premise.operands
        for name in (atom.subject, atom.reference)
    }
    if named != set(problem.objects):
        raise ValueError("plain scenes require every object to occur in a premise")
    objects = tuple(sorted(problem.objects))
    query = problem.query
    if not isinstance(query, DirectionQuery):
        query = replace(
            query, candidates=tuple(o for o in objects if o != query.reference)
        )
    return replace(problem, objects=objects, query=query)


def require_scene_scope(problem):
    if all_scene_problem(problem) != problem:
        raise ValueError("plain prompt requires canonical all-scene scope")


def parse_row(row):
    """Dispatch explicitly; an old scoped row must never acquire new semantics."""
    from spatial.v2.entailment_contracts import VERSION as INTERFACE
    from spatial.v2.entailment_contracts import parse_prompt as old

    if row.get("interface") != INTERFACE:
        raise ValueError("row is not entailment-only")
    if row.get("prompt_version") is None:
        if row.get("scope_policy") is not None or row.get("display_names") is not None:
            raise ValueError("scene metadata requires the plain protocol")
        return old(row["prompt"])
    if row["prompt_version"] not in PLAIN_VERSIONS or row.get("scope_policy") != SCOPE:
        raise ValueError("unknown or mismatched scene protocol")
    if (
        row["prompt_version"] == SHORT_VERSION
        and row.get("contract") != Contract.UNIQUE.value
    ):
        raise ValueError("short questions require the Unique contract")
    return parse_prompt(
        row["prompt"],
        display_names=row.get("display_names"),
        prompt_version=row["prompt_version"],
    )


def _names(problem, display_names):
    names = {o: o for o in problem.objects} if display_names is None else display_names
    if (
        set(names) != set(problem.objects)
        or len(set(names.values())) != len(names)
        or any(
            not isinstance(v, str) or not v or not v.isprintable()
            for v in names.values()
        )
    ):
        raise ValueError("display names must be a printable bijection")
    return names


def display_references(pair, display_names):
    """Substitute external aliases for display without changing proof data."""
    if display_names is None:
        return pair
    if any(re.fullmatch(r"Entity\d+", name) is None for name in display_names):
        raise ValueError("reference display mapping requires reserved external aliases")
    alternatives = "|".join(
        re.escape(name) for name in sorted(display_names, key=len, reverse=True)
    )
    pattern = rf"(?<![A-Za-z0-9_])(?:{alternatives})(?![A-Za-z0-9_])"
    return {
        style: re.sub(pattern, lambda m: display_names[m.group()], text)
        for style, text in pair.items()
    }


def render_prompt(
    problem,
    menu,
    contract,
    *,
    display_names=None,
    scene_paragraph=None,
    layout="paragraph",
    prompt_version=VERSION,
):
    require_scene_scope(problem)
    validate_menu(problem, menu)
    contract = Contract(contract)
    if prompt_version not in PLAIN_VERSIONS:
        raise ValueError("unknown plain prompt version")
    if prompt_version == SHORT_VERSION and (
        contract is not Contract.UNIQUE or layout != "paragraph"
    ):
        raise ValueError("short questions require Unique and paragraph layout")
    templates = SHORT_QUESTIONS if prompt_version == SHORT_VERSION else QUESTIONS
    names = _names(problem, display_names)
    query = problem.query
    kind = type(query).__name__.removesuffix("Query").lower()
    fields = {"reference": names[query.reference]}
    if isinstance(query, DirectionQuery):
        fields["target"] = names[query.target]
    else:
        fields["direction"] = next(iter(query.directions)).value.lower()
    sentences = [
        f"{names[a.subject]} is to the {next(iter(a.allowed)).value} "
        f"of {names[a.reference]}."
        for a in problem.premise.operands
    ]
    if layout not in ("paragraph", "lines"):
        raise ValueError("unknown premise layout")
    if scene_paragraph is not None:
        if layout != "paragraph":
            raise ValueError("source paragraph requires paragraph layout")
        body = scene_paragraph
        declaration = re.match(r"(.+?) is in the map\. ", body)
        if declaration:
            if declaration.group(1) not in names.values():
                raise ValueError("source declaration names an unknown object")
            body = body[declaration.end() :]
        if body != " ".join(sentences):
            raise ValueError("source paragraph differs from spatial premises")
    scene = (
        [scene_paragraph or " ".join(sentences), ""]
        if layout == "paragraph"
        else sentences
    )
    lines = [
        INTRO,
        *scene,
        templates[kind, contract].format(**fields),
        *(
            f"{letter}. {names[value] if isinstance(query, WhichQuery) else value}"
            for letter, value in zip("ABCD", menu)
        ),
        "E. Cannot be determined",
        "F. No complete entailed answer can be formed from A-D."
        if contract is Contract.EXHAUSTIVE
        else "F. None of the above",
    ]
    return "\n".join(lines)


def _parse_question(text, prompt_version):
    if prompt_version not in PLAIN_VERSIONS:
        raise ValueError("unknown plain prompt version")
    templates = SHORT_QUESTIONS if prompt_version == SHORT_VERSION else QUESTIONS
    for (kind, contract), template in templates.items():
        pattern = re.escape(template)
        for field in ("target", "reference", "direction"):
            pattern = pattern.replace(re.escape("{" + field + "}"), f"(?P<{field}>.+?)")
        match = re.fullmatch(pattern, text)
        if match:
            return kind, contract, match.groupdict()
    raise ValueError("question does not match the plain entailment protocol")


def parse_prompt(text, *, display_names=None, prompt_version=VERSION):
    lines = text.splitlines()
    if len(lines) < 9 or lines[0] != INTRO:
        raise ValueError("invalid plain spatial scene")
    kind, contract, fields = _parse_question(lines[-7], prompt_version)
    inverse = {v: k for k, v in (display_names or {}).items()}

    def entity(name):
        if display_names is not None and name not in inverse:
            raise ValueError("unknown displayed scene object")
        return inverse.get(name, name)

    scene_lines = lines[1:-7]
    paragraph = len(scene_lines) == 2 and scene_lines[-1] == ""
    scene_paragraph = scene_lines[0] if paragraph else None
    declaration = None
    if paragraph:
        body = scene_paragraph
        match = re.match(r"(.+?) is in the map\. ", body)
        if match:
            declaration = entity(match.group(1))
            body = body[match.end() :]
        scene_lines = re.split(r"(?<=\.) ", body)
    atoms, objects = [], set()
    alternatives = "|".join(d.value for d in Direction)
    for line in scene_lines:
        match = re.fullmatch(rf"(.+?) is to the ({alternatives}) of (.+)\.", line)
        if not match:
            raise ValueError("invalid exact spatial premise")
        subject, direction, reference = match.groups()
        subject, reference = entity(subject), entity(reference)
        objects.update((subject, reference))
        atoms.append(direction_constraint(subject, reference, Direction(direction)))
    objects = tuple(sorted(objects))
    if declaration is not None and declaration not in objects:
        raise ValueError("source declaration names an unknown object")
    reference = entity(fields["reference"])
    if kind == "direction":
        query = DirectionQuery(
            entity(fields["target"]), reference, frozenset(Direction)
        )
    else:
        directions = frozenset((Direction(fields["direction"].capitalize()),))
        candidates = tuple(o for o in objects if o != reference)
        query = (WhichQuery if kind == "which" else CountQuery)(
            directions, reference, candidates
        )
    problem = SpatialProblem(objects, And(tuple(atoms)), query)
    menu = []
    for letter, line in zip("ABCD", lines[-6:-2]):
        if not line.startswith(letter + ". "):
            raise ValueError("invalid four-option menu")
        value = line[3:]
        menu.append(
            int(value)
            if kind == "count"
            else entity(value)
            if kind == "which"
            else value
        )
    menu = tuple(menu)
    if (
        render_prompt(
            problem,
            menu,
            contract,
            display_names=display_names,
            scene_paragraph=scene_paragraph,
            layout="paragraph" if paragraph else "lines",
            prompt_version=prompt_version,
        )
        != text
    ):
        raise ValueError("plain prompt does not round-trip exactly")
    return problem, menu, contract
