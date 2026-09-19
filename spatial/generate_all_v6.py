"""v6 synthetic SFT generator.

Proposes a map + question. Gold is only whatever
``eval.spatial_solver.SpatialSolver.grade`` accepts. Illegal gold
(errors, empty, 4-letter) is dropped. Optional ``inject_conflict``
adds a reversing sentence so cycles can appear; the solver still labels.

Leave finetune/generate_all.py as the pre-cleanup generator.
See docs/symbolic-cot.md.
"""
import json
import itertools
import random
import sys
from collections import defaultdict
from pathlib import Path

import typer

from spatial_solver import SpatialSolver  # noqa: E402

_SOLVER = SpatialSolver()

# Preset entity name library — no duplicates allowed
ENTITIES = [
    "Police Station",
    "Library",
    "Coffee Shop",
    "Supermarket",
    "Hospital",
    "Post Office",
    "Fire Department",
    "Museum",
    "Park",
    "Gas Station",
    "High School",
    "Cinema",
    "University",
    "City Hall",
    "Zoo",
    "Shopping Mall",
    "Bakery",
    "Church",
    "Bank",
    "Pharmacy",
]

# Safety check: ensure no duplicate entity names at import time
assert len(ENTITIES) == len(set(ENTITIES)), "ENTITIES list contains duplicates!"

ALL_DIRECTIONS = [
    "North", "South", "East", "West",
    "Northeast", "Northwest", "Southeast", "Southwest",
]

# Type 0 fifth choice. Both axes unknown ⇒ none of the four compounds is
# proven, so gold is E, not A,B,C,D.
UNDETERMINED_OPTION = "Cannot be determined"
NONE_OF_OPTIONS_CANON = "None of the Options"
NONE_OF_OPTIONS_PHRASES = [
    "None of the Options",
    "None of these options",
    "None of the given options",
    "Not among the options",
    "No listed option is correct",
]


class AxisGraph:
    """Single-axis (X-axis or Y-axis) directed acyclic graph (DAG) manager"""

    def __init__(self):
        self.nodes = set()
        self.edges = set()  # (A, B) means A < B

    def add_relation(self, a, b):
        self.nodes.add(a)
        self.nodes.add(b)
        self.edges.add((a, b))

    def get_transitive_closure(self):
        """Compute the global transitive closure (all implied relations)"""
        closure = set(self.edges)
        added = True
        while added:
            added = False
            new_edges = set()
            for a, b in closure:
                for c, d in closure:
                    if b == c and (a, d) not in closure:
                        new_edges.add((a, d))
                        added = True
            closure.update(new_edges)
        return closure

    def format_state(self, active_nodes=None):
        """Format the current graph state as SFT text (incremental activation strategy)"""
        target_nodes = active_nodes if active_nodes is not None else self.nodes
        if not target_nodes:
            return "Empty"

        global_closure = self.get_transitive_closure()
        sub_closure = {
            (a, b) for a, b in global_closure if a in target_nodes and b in target_nodes
        }

        sub_reduction = set(sub_closure)
        for a, b in list(sub_reduction):
            for c in target_nodes:
                if (a, c) in sub_closure and (c, b) in sub_closure:
                    if (a, b) in sub_reduction:
                        sub_reduction.remove((a, b))

        if not sub_reduction:
            return ", ".join(sorted(list(target_nodes)))

        preds = defaultdict(set)
        succs = defaultdict(set)
        for a, b in sub_closure:
            preds[b].add(a)
            succs[a].add(b)

        groups = defaultdict(list)
        for node in target_nodes:
            key = (frozenset(preds[node]), frozenset(succs[node]))
            groups[key].append(node)

        group_names = {}
        node_to_group = {}
        for key, members in groups.items():
            sorted_members = sorted(members)
            if len(sorted_members) == 1:
                name = sorted_members[0]
            else:
                name = "{" + ", ".join(sorted_members) + "}"
            group_names[key] = name
            for m in members:
                node_to_group[m] = name

        group_edges = set()
        for a, b in sub_reduction:
            ga = node_to_group[a]
            gb = node_to_group[b]
            if ga != gb:
                group_edges.add((ga, gb))

        if not group_edges:
            return ", ".join(sorted(set(node_to_group.values())))

        next_node_map = defaultdict(list)
        in_degree = defaultdict(int)
        nodes_in_edges = set()

        for u, v in group_edges:
            next_node_map[u].append(v)
            in_degree[v] += 1
            if u not in in_degree:
                in_degree[u] = 0
            nodes_in_edges.add(u)
            nodes_in_edges.add(v)

        starts = [node for node, deg in in_degree.items() if deg == 0]
        chains = []

        def build_path(current_node, current_path):
            neighbors = sorted(next_node_map[current_node])
            if not neighbors:
                chains.append(" < ".join(current_path))
                return
            for nxt in neighbors:
                build_path(nxt, current_path + [nxt])

        for start in sorted(starts):
            build_path(start, [start])

        isolated = set(node_to_group.values()) - nodes_in_edges
        for iso in sorted(isolated):
            chains.append(iso)

        return ", ".join(sorted(list(set(chains))))


# ---------------------------------------------------------------------------
# Helper utilities for Type 1 & Type 2 questions
# ---------------------------------------------------------------------------

def get_entities_in_direction(ref, direction, x_closure, y_closure, all_entities):
    """Return a sorted list of entities *definitively* in ``direction`` of ``ref``.

    X-axis convention: (a, b) in x_closure  ⟹  a < b  (a is West of b).
    Y-axis convention: (a, b) in y_closure  ⟹  a < b  (a is South of b).
    """
    entities = []
    for entity in all_entities:
        if entity == ref:
            continue

        is_east = (ref, entity) in x_closure and (entity, ref) not in x_closure
        is_west = (entity, ref) in x_closure and (ref, entity) not in x_closure
        is_north = (ref, entity) in y_closure and (entity, ref) not in y_closure
        is_south = (entity, ref) in y_closure and (ref, entity) not in y_closure

        match = False
        if direction == "East":
            match = is_east
        elif direction == "West":
            match = is_west
        elif direction == "North":
            match = is_north
        elif direction == "South":
            match = is_south
        elif direction == "Northeast":
            match = is_east and is_north
        elif direction == "Northwest":
            match = is_west and is_north
        elif direction == "Southeast":
            match = is_east and is_south
        elif direction == "Southwest":
            match = is_west and is_south

        if match:
            entities.append(entity)

    return sorted(entities)


def _direction_x_component(direction):
    """Return the X-axis component ('East'/'West') or None."""
    if direction in ("East", "Northeast", "Southeast"):
        return "East"
    if direction in ("West", "Northwest", "Southwest"):
        return "West"
    return None


def _direction_y_component(direction):
    """Return the Y-axis component ('North'/'South') or None."""
    if direction in ("North", "Northeast", "Northwest"):
        return "North"
    if direction in ("South", "Southeast", "Southwest"):
        return "South"
    return None


def _axis_fact(ref, target, closure, positive, negative):
    """How ``target`` sits relative to ``ref`` on one axis.

    ``closure``: (a, b) means a < b. If both directions exist, that is a
    cycle — not a proof.
    """
    fwd = (ref, target) in closure
    back = (target, ref) in closure
    if fwd and back:
        return None, f"both {ref} < {target} and {target} < {ref} (contradiction)"
    if fwd:
        return positive, f"{ref} < {target}"
    if back:
        return negative, f"{target} < {ref}"
    return None, f"neither {ref} < {target} nor {target} < {ref}"


def _axis_determined(a, b, closure) -> bool:
    fwd = (a, b) in closure
    back = (b, a) in closure
    if fwd and back:
        return False
    return fwd or back


def _make_conflict_sentence(sent, mode: str) -> dict:
    """Append a sentence that fights ``sent``.

    full: swap endpoints, same compound → cycle on both axes.
    one_axis: swap endpoints and flip N/S → X cycles, Y still agrees.
    """
    if mode == "one_axis":
        flip_y = "South" if sent["dir_y"] == "North" else "North"
        direction = f"{flip_y}{sent['dir_x'].lower()}"
        return {
            "text": f"The {sent['b']} is to the {direction} of the {sent['a']}.",
            "a": sent["b"],
            "b": sent["a"],
            "dir": direction,
            "dir_x": sent["dir_x"],
            "dir_y": flip_y,
        }
    direction = sent["dir"]
    return {
        "text": f"The {sent['b']} is to the {direction} of the {sent['a']}.",
        "a": sent["b"],
        "b": sent["a"],
        "dir": direction,
        "dir_x": sent["dir_x"],
        "dir_y": sent["dir_y"],
    }


def _special_gold(grade) -> bool:
    """True iff the sole gold option is undetermined / none-of-options (any letter)."""
    inns = [(let, val) for let, val, inn, _ in grade.verdicts if inn]
    if len(inns) != 1:
        return False
    val = inns[0][1]
    return SpatialSolver.is_none_of_above(val) or SpatialSolver.is_undetermined_option(val)


def _target_ok(grade, question_type, target_num_answers) -> bool:
    n = len(grade.letters)
    if question_type == 0:
        if target_num_answers is None:
            return True
        if target_num_answers == 1:
            return n == 1 and not _special_gold(grade)
        if target_num_answers == 2:
            return n == 2
        if target_num_answers == 0:
            return _special_gold(grade)
        return False
    if question_type == 1:
        if target_num_answers == 0:
            return _special_gold(grade)
        if target_num_answers is None:
            return 1 <= n <= 4
        return n == target_num_answers
    if question_type == 2:
        if target_num_answers == 0:
            return _special_gold(grade)
        return n == 1 and not _special_gold(grade)
    return False


def _none_of_options_text(shuffle_phrase: bool) -> str:
    if shuffle_phrase:
        return random.choice(NONE_OF_OPTIONS_PHRASES)
    return NONE_OF_OPTIONS_CANON


def _fifth_option(*, gold_is_none: bool, shuffle_none_phrase: bool) -> str:
    """Fifth MCQ slot. Only force None of the Options when that is gold.

    If A–D already contain the theorem, the fifth slot is just a distractor
    (Cannot be determined or None of the Options).
    """
    if gold_is_none:
        return _none_of_options_text(shuffle_none_phrase)
    return random.choice(
        [UNDETERMINED_OPTION, _none_of_options_text(shuffle_none_phrase)]
    )


def _attach_specials(choices, specials, shuffle_special: bool) -> list:
    """Content options plus specials. Shuffle mixes specials off the last letters."""
    opts = list(choices) + list(specials)
    if shuffle_special:
        random.shuffle(opts)
    return opts


def _attach_special(choices, special, shuffle_special: bool) -> list:
    return _attach_specials(choices, [special], shuffle_special)


def _pack_sample(system_prompt, user_prompt, steps_text, prefix_lines, question_type, target_num_answers):
    """Label with the solver. Drop anything it will not accept."""
    grade = _SOLVER.grade(user_prompt)
    if not grade.accept:
        return None
    if len(grade.options) != 5:
        return None
    if not _target_ok(grade, question_type, target_num_answers):
        return None
    deduction = list(prefix_lines) + ["**Options**:"]
    for letter, val, inn, reason in grade.verdicts:
        mark = "In" if inn else "Out"
        deduction.append(f"- {letter}. {val} — {reason}. {mark}.")
    thinking_content = f"{steps_text[0]}\n\n"
    for step in steps_text[1:]:
        thinking_content += f"{step}\n\n"
    thinking_content += "### Final Deduction\n"
    thinking_content += "\n".join(deduction) + "\n"
    answer_str = grade.pretty or grade.raw.replace(",", ", ")
    final_deduction = (
        f"<think>\n{thinking_content}\n</think>\nAnswer: {answer_str}"
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": final_deduction},
        ]
    }


def _type0_option_verdict(opt, x_comp, y_comp):
    """Accept/reject one Type 0 option from derived axis components."""
    both_unknown = x_comp is None and y_comp is None
    if opt == UNDETERMINED_OPTION:
        if both_unknown:
            return True, "neither axis is derived, so no compound is proven"
        known = [c for c in (x_comp, y_comp) if c]
        verb = "are" if len(known) == 2 else "is"
        return False, (
            f"{' and '.join(known)} {verb} derived, so the direction "
            f"is not fully unknown"
        )

    need_x = _direction_x_component(opt)
    need_y = _direction_y_component(opt)
    if need_x is None and need_y is None:
        return False, "not a direction derived from the axes"

    if x_comp and need_x and need_x != x_comp:
        return False, f"needs {need_x}, but X-axis is {x_comp}"
    if y_comp and need_y and need_y != y_comp:
        return False, f"needs {need_y}, but Y-axis is {y_comp}"

    if both_unknown:
        return False, "neither axis is derived, so this direction is not proven"

    # Cardinals are distractors; gold is a compound or E.
    if need_x is None or need_y is None:
        return False, "only one axis; live answers are two-axis compounds (or E)"

    if x_comp and y_comp:
        if need_x == x_comp and need_y == y_comp:
            return True, f"{x_comp} and {y_comp} both derived"
        return False, f"derived direction is {y_comp}{x_comp.lower()}"

    if y_comp and need_y == y_comp:
        return True, f"{y_comp} is derived; East/West unknown, so this remains possible"
    if x_comp and need_x == x_comp:
        return True, f"{x_comp} is derived; North/South unknown, so this remains possible"
    return False, "does not match the derived axis"


def _type1_entity_verdict(entity, ref, direction, x_closure, y_closure):
    """Prove whether ``entity`` is definitely in ``direction`` of ``ref``."""
    need_x = _direction_x_component(direction)
    need_y = _direction_y_component(direction)
    x_comp, x_fact = _axis_fact(ref, entity, x_closure, "East", "West")
    y_comp, y_fact = _axis_fact(ref, entity, y_closure, "North", "South")
    parts = []
    ok = True
    if need_x:
        if x_comp == need_x:
            parts.append(f"{x_fact} → {need_x}")
        elif x_comp:
            ok = False
            parts.append(f"{x_fact} → {x_comp}, not {need_x}")
        else:
            ok = False
            parts.append(f"{x_fact} → {need_x} not derived")
    if need_y:
        if y_comp == need_y:
            parts.append(f"{y_fact} → {need_y}")
        elif y_comp:
            ok = False
            parts.append(f"{y_fact} → {y_comp}, not {need_y}")
        else:
            ok = False
            parts.append(f"{y_fact} → {need_y} not derived")
    if not need_x and not need_y:
        return False, "not a known direction"
    return ok, "; ".join(parts)


def _build_direction_reasoning(ref, direction, entities_in_dir,
                               x_closure, y_closure, all_mentioned):
    """Build reasoning lines for the Final Deduction of Type 1 / Type 2."""
    x_comp = _direction_x_component(direction)
    y_comp = _direction_y_component(direction)
    lines = []
    lines.append(f"**Reference**: {ref}")
    lines.append(f"**Direction Query**: {direction}")

    if x_comp and not y_comp:
        # Pure East / West
        x_label = f"{ref} < Entity" if x_comp == "East" else f"Entity < {ref}"
        x_ents = get_entities_in_direction(ref, x_comp, x_closure, y_closure, all_mentioned)
        lines.append(
            f"**X-Axis Analysis**: Entities to the {x_comp} of the {ref} "
            f"({x_label} on X-axis): "
            f"{', '.join(x_ents) if x_ents else 'None'}"
        )
    elif y_comp and not x_comp:
        # Pure North / South
        y_label = f"{ref} < Entity" if y_comp == "North" else f"Entity < {ref}"
        y_ents = get_entities_in_direction(ref, y_comp, x_closure, y_closure, all_mentioned)
        lines.append(
            f"**Y-Axis Analysis**: Entities to the {y_comp} of the {ref} "
            f"({y_label} on Y-axis): "
            f"{', '.join(y_ents) if y_ents else 'None'}"
        )
    else:
        # Intercardinal — show both axes then intersection
        x_ents = get_entities_in_direction(ref, x_comp, x_closure, y_closure, all_mentioned)
        y_ents = get_entities_in_direction(ref, y_comp, x_closure, y_closure, all_mentioned)
        x_label = f"{ref} < Entity" if x_comp == "East" else f"Entity < {ref}"
        y_label = f"{ref} < Entity" if y_comp == "North" else f"Entity < {ref}"
        lines.append(
            f"**X-Axis Analysis**: Entities to the {x_comp} of the {ref} "
            f"({x_label} on X-axis): "
            f"{', '.join(x_ents) if x_ents else 'None'}"
        )
        lines.append(
            f"**Y-Axis Analysis**: Entities to the {y_comp} of the {ref} "
            f"({y_label} on Y-axis): "
            f"{', '.join(y_ents) if y_ents else 'None'}"
        )
        lines.append(
            f"**Entities in {direction} (intersection)**: "
            f"{', '.join(entities_in_dir) if entities_in_dir else 'None'}"
        )

    return lines


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are an advanced spatial reasoning agent. Process the spatial "
    "relations step-by-step.\nFirst, initialize by detecting all entities "
    "in the text to establish the global scope. \nThen, step-by-step, "
    "extract relations and update the spatial state for the X-axis "
    "(West to East) and Y-axis (South to North), tracking ONLY the "
    "entities that have been mentioned so far.\nRules for State "
    "Representation:\n1. Use \"<\" for strict ordering (e.g., A < B means "
    "A is West/South of B).\n2. Group topologically equivalent entities "
    "using \"{}\" (e.g., {A, B} < C).\n3. Merge relations into maximal "
    "chains to represent the spatial topology clearly.\n4. Keep isolated "
    "active entities separated by commas until they are connected.\n\n"
    "CRITICAL RULE FOR FINAL DEDUCTION:\nRead the final X-State and "
    "Y-State. For the queried pair, derive each axis from whether "
    "A < B is in that state. A two-axis direction is proven only if "
    "both components are derived. If exactly one axis is derived, the "
    "remaining compounds that match it are the only possible answers. "
    "If neither axis is derived, select \"Cannot be determined\". Then "
    "accept or reject each option from those facts."
)


def generate_sample(num_entities=5, num_sentences=6, target_num_answers=None,
                    question_type=0, inject_conflict=None,
                    incomplete_pair=False, omit_live=False,
                    shuffle_special=False, shuffle_none_phrase=False,
                    system_prompt=None):
    """Generate a single SFT training sample.

    Args:
        num_entities: Number of entities to use.
        num_sentences: Number of relation sentences to generate.
        target_num_answers: Desired number of correct answers.
            Type 0 — 1 (both axes proven), 2 (one axis proven), or 0
            (both unknown: gold is E, "Cannot be determined").
            Type 1 — 1 or 2 proven entity options among A–D.
            Type 2 — ignored (count question, always exactly 1 correct count).
            If None, Type 0 accepts 1 / 2 / E; Type 1 accepts 1 or 2.
        inject_conflict: None, "full" (both-axis cycle), or "one_axis"
            (X cycles, Y still agrees). Extra sentence is appended; the
            solver still decides gold.
        incomplete_pair: Type 0, one axis known: put only one of the two
            remaining compounds on the menu (e.g. Southeast but not
            Southwest). Gold is Cannot be determined.
        omit_live: Type 0: omit every live compound from A–D. Gold is
            None of the Options (a theorem exists, it is just not listed).
        shuffle_special: mix the special option into A–E instead of
            always putting it last as E.
        shuffle_none_phrase: pick a paraphrase of "None of the Options"
            so the model cannot key on one string.
        question_type:
            0 — original direction question
            1 — which-entity question ("Which object …")
            2 — count question  ("How many objects …")
            Numbering matches eval/clean_v6.py.

    Returns:
        A sample dict, or None if no suitable question can be formed.
    """
    # ==================== Phase 1: Build world ====================
    selected_entities = random.sample(ENTITIES, num_entities)
    coords = {}

    for ent in selected_entities:
        while True:
            x, y = random.randint(0, 100), random.randint(0, 100)
            if not any(c[0] == x or c[1] == y for c in coords.values()):
                coords[ent] = (x, y)
                break

    sentences = []
    pairs = list(itertools.combinations(selected_entities, 2))
    random.shuffle(pairs)

    for a, b in pairs[:num_sentences]:
        if a == b:
            continue

        dx = coords[a][0] - coords[b][0]
        dy = coords[a][1] - coords[b][1]

        dir_x = "East" if dx > 0 else "West"
        dir_y = "North" if dy > 0 else "South"

        direction = f"{dir_y}{dir_x.lower()}"
        sentences.append(
            {
                "text": f"The {a} is to the {direction} of the {b}.",
                "a": a,
                "b": b,
                "dir": direction,
                "dir_x": dir_x,
                "dir_y": dir_y,
            }
        )

    if not sentences:
        return None

    conflict_pair = None
    if inject_conflict in ("full", "one_axis"):
        extra = _make_conflict_sentence(random.choice(sentences), inject_conflict)
        sentences.append(extra)
        conflict_pair = (extra["a"], extra["b"])

    # ==================== Phase 2: Incremental reasoning ====================
    x_graph = AxisGraph()
    y_graph = AxisGraph()
    steps_text = []

    all_mentioned_entities = set()
    for sent in sentences:
        all_mentioned_entities.add(sent["a"])
        all_mentioned_entities.add(sent["b"])

    for ent in all_mentioned_entities:
        x_graph.nodes.add(ent)
        y_graph.nodes.add(ent)

    init_str = "### Initialization\n"
    init_str += (
        f"**Entities Detected**: {', '.join(sorted(list(all_mentioned_entities)))}\n"
    )
    init_str += f"**Initial X-State**: {x_graph.format_state()}\n"
    init_str += f"**Initial Y-State**: {y_graph.format_state()}\n"
    steps_text.append(init_str)

    active_entities = set()

    for i, sent in enumerate(sentences):
        a, b = sent["a"], sent["b"]
        active_entities.add(a)
        active_entities.add(b)

        if sent["dir_x"] == "East":
            x_graph.add_relation(b, a)
            x_ext = f"{b} < {a}"
        else:
            x_graph.add_relation(a, b)
            x_ext = f"{a} < {b}"

        if sent["dir_y"] == "North":
            y_graph.add_relation(b, a)
            y_ext = f"{b} < {a}"
        else:
            y_graph.add_relation(a, b)
            y_ext = f"{a} < {b}"

        step_str = f"### Step {i + 1}\n"
        step_str += f'**Sentence**: "{sent["text"]}"\n'
        step_str += f"**X-Extraction**: {x_ext}\n"
        step_str += f"**Y-Extraction**: {y_ext}\n"
        step_str += f"**X-State**: {x_graph.format_state(active_entities)}\n"
        step_str += f"**Y-State**: {y_graph.format_state(active_entities)}\n"
        steps_text.append(step_str)

    # ==================== Phase 3: Generate QA ====================
    x_closure = x_graph.get_transitive_closure()
    y_closure = y_graph.get_transitive_closure()

    mentioned_pairs = [
        (a, b) for a, b in pairs
        if a in all_mentioned_entities and b in all_mentioned_entities and a != b
    ]

    if not system_prompt:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    option_letters = ["A", "B", "C", "D"]

    if question_type == 0 and target_num_answers is not None and target_num_answers not in (0, 1, 2):
        return None
    if question_type == 1 and target_num_answers is not None and target_num_answers not in (0, 1, 2, 3, 4):
        return None

    # ------------------------------------------------------------------
    # TYPE 0 — Original direction question
    # ------------------------------------------------------------------
    if question_type == 0:
        if not mentioned_pairs:
            return None

        if conflict_pair is not None:
            target, ref = conflict_pair
        elif incomplete_pair or omit_live:
            candidate_pairs = []
            for a, b in mentioned_pairs:
                x_ok = _axis_determined(a, b, x_closure)
                y_ok = _axis_determined(a, b, y_closure)
                if incomplete_pair and (x_ok ^ y_ok):
                    candidate_pairs.append((a, b))
                elif omit_live and (x_ok or y_ok):
                    candidate_pairs.append((a, b))
            if not candidate_pairs:
                return None
            target, ref = random.choice(candidate_pairs)
        elif target_num_answers is not None:
            candidate_pairs = []
            for a, b in mentioned_pairs:
                x_determined = _axis_determined(a, b, x_closure)
                y_determined = _axis_determined(a, b, y_closure)

                if x_determined and y_determined:
                    num_answers = 1
                elif x_determined or y_determined:
                    num_answers = 2
                else:
                    num_answers = 0  # undetermined; gold is E, not A,B,C,D

                if num_answers == target_num_answers:
                    candidate_pairs.append((a, b))

            if not candidate_pairs:
                return None
            target, ref = random.choice(candidate_pairs)
        else:
            target, ref = random.choice(mentioned_pairs)

        x_comp, x_fact = _axis_fact(ref, target, x_closure, "East", "West")
        y_comp, y_fact = _axis_fact(ref, target, y_closure, "North", "South")

        xs = [x_comp] if x_comp else ["East", "West"]
        ys = [y_comp] if y_comp else ["North", "South"]
        possible_dirs = [f"{y}{x.lower()}" for y in ys for x in xs]

        all_dir_options = [
            "North", "South", "East", "West",
            "Northeast", "Northwest", "Southeast", "Southwest",
        ]
        if incomplete_pair:
            if len(possible_dirs) != 2:
                return None
            kept = random.choice(possible_dirs)
            omitted = [d for d in possible_dirs if d != kept]
            choices = [kept]
            while len(choices) < 4:
                d = random.choice(all_dir_options)
                if d not in choices and d not in omitted:
                    choices.append(d)
            random.shuffle(choices)
        elif omit_live:
            if not possible_dirs:
                return None
            choices = []
            while len(choices) < 4:
                d = random.choice(all_dir_options)
                if d not in choices and d not in possible_dirs:
                    choices.append(d)
            random.shuffle(choices)
        else:
            choices = list(possible_dirs)
            while len(choices) < 4:
                d = random.choice(all_dir_options)
                if d not in choices:
                    choices.append(d)
            random.shuffle(choices)
        listed_live = [d for d in possible_dirs if d in choices]
        if not possible_dirs or incomplete_pair:
            special = UNDETERMINED_OPTION
        elif possible_dirs and not listed_live:
            special = _none_of_options_text(shuffle_none_phrase)
        else:
            # A–D already hold the gold; fifth slot is any special distractor.
            special = _fifth_option(
                gold_is_none=False,
                shuffle_none_phrase=shuffle_none_phrase,
            )
        options = _attach_special(choices, special, shuffle_special)
        if len(options) != 5:
            return None
        if (
            not incomplete_pair
            and (x_comp is None) != (y_comp is None)
            and any(d not in options for d in possible_dirs)
        ):
            return None
        type0_letters = ["A", "B", "C", "D", "E"][: len(options)]

        if x_comp and y_comp:
            derived_line = (
                f"**Derived direction**: {y_comp}{x_comp.lower()} "
                f"(both axes proven)."
            )
        elif x_comp or y_comp:
            known = x_comp or y_comp
            unk = "North/South" if x_comp else "East/West"
            derived_line = (
                f"**Derived direction**: {known} proven, {unk} unknown "
                f"→ must be {' or '.join(possible_dirs)}."
            )
        else:
            derived_line = (
                "**Derived direction**: none (both axes unknown"
                + (" / contradiction" if "contradiction" in x_fact + y_fact else "")
                + ")."
            )

        x_line = (
            f"**X-axis**: {x_fact} → {target} is {x_comp} of {ref}."
            if x_comp else
            f"**X-axis**: {x_fact} → East/West unknown."
        )
        y_line = (
            f"**Y-axis**: {y_fact} → {target} is {y_comp} of {ref}."
            if y_comp else
            f"**Y-axis**: {y_fact} → North/South unknown."
        )

        options_text = ", ".join(
            [f"{type0_letters[i]}. {options[i]}" for i in range(len(options))]
        )
        user_prompt = "Consider a map with multiple locations:\n\n"
        user_prompt += " ".join([s["text"] for s in sentences])
        user_prompt += (
            f"\n\nQuestion: In which direction is the {target} relative to "
            f"the {ref}? Available options: {options_text}"
        )
        prefix = [
            f"**Target**: {target}",
            f"**Reference**: {ref}",
            f"**X-State**: {x_graph.format_state(active_entities)}",
            f"**Y-State**: {y_graph.format_state(active_entities)}",
            x_line,
            y_line,
            derived_line,
        ]
        return _pack_sample(
            system_prompt, user_prompt, steps_text, prefix,
            question_type, target_num_answers,
        )

    # ------------------------------------------------------------------
    # TYPE 2 — Count question
    # ------------------------------------------------------------------
    elif question_type == 2:
        ref_candidates = sorted(all_mentioned_entities)
        random.shuffle(ref_candidates)

        found = False
        ref = direction = None
        entities_in_dir = []

        for _ref in ref_candidates:
            dir_list = list(ALL_DIRECTIONS)
            random.shuffle(dir_list)
            for _dir in dir_list:
                _ents = get_entities_in_direction(
                    _ref, _dir, x_closure, y_closure, all_mentioned_entities
                )
                # Accept the first valid combination
                ref = _ref
                direction = _dir
                entities_in_dir = _ents
                found = True
                break
            if found:
                break

        if not found:
            return None

        correct_count = len(entities_in_dir)

        max_count = len(all_mentioned_entities) - 1
        omit_true = target_num_answers == 0
        option_values = [] if omit_true else [correct_count]
        wrong_pool = [i for i in range(0, max_count + 1) if i != correct_count]
        random.shuffle(wrong_pool)
        for v in wrong_pool:
            if len(option_values) >= 4:
                break
            option_values.append(v)
        extra = max_count + 1
        while len(option_values) < 4:
            if extra not in option_values:
                option_values.append(extra)
            extra += 1
        random.shuffle(option_values)
        type2_menu = _attach_special(
            option_values,
            _fifth_option(
                gold_is_none=omit_true,
                shuffle_none_phrase=shuffle_none_phrase,
            ),
            shuffle_special,
        )
        if len(type2_menu) != 5:
            return None
        type2_letters = ["A", "B", "C", "D", "E"]

        reasoning_lines = [
            f"**X-State**: {x_graph.format_state(active_entities)}",
            f"**Y-State**: {y_graph.format_state(active_entities)}",
        ]
        reasoning_lines.extend(
            _build_direction_reasoning(
                ref, direction, entities_in_dir,
                x_closure, y_closure, all_mentioned_entities,
            )
        )
        reasoning_lines.append(
            f"**Count**: |intersection| = {correct_count}."
        )
        options_text = ", ".join(
            [f"{type2_letters[i]}. {type2_menu[i]}" for i in range(5)]
        )
        user_prompt = "Consider a map with multiple locations:\n\n"
        user_prompt += " ".join([s["text"] for s in sentences])
        user_prompt += (
            f"\n\nQuestion: How many objects are in the {direction} of the "
            f"{ref}? Available options: {options_text}"
        )
        return _pack_sample(
            system_prompt, user_prompt, steps_text, reasoning_lines,
            question_type, target_num_answers,
        )

    # ------------------------------------------------------------------
    # TYPE 1 — Which-entity question
    # ------------------------------------------------------------------
    elif question_type == 1:
        min_correct = 1 if target_num_answers in (None, 0) else target_num_answers

        ref_candidates = sorted(all_mentioned_entities)
        random.shuffle(ref_candidates)

        found = False
        ref = direction = None
        entities_in_dir = []
        entities_not_in_dir = []

        for _ref in ref_candidates:
            dir_list = list(ALL_DIRECTIONS)
            random.shuffle(dir_list)
            for _dir in dir_list:
                _ents = get_entities_in_direction(
                    _ref, _dir, x_closure, y_closure, all_mentioned_entities
                )
                _not_ents = [
                    e for e in sorted(all_mentioned_entities)
                    if e != _ref and e not in _ents
                ]
                need_wrong = 4 if target_num_answers == 0 else (4 - min_correct)
                if (len(_ents) >= min_correct
                        and len(_not_ents) >= need_wrong):
                    ref = _ref
                    direction = _dir
                    entities_in_dir = _ents
                    entities_not_in_dir = _not_ents
                    found = True
                    break
            if found:
                break

        if not found:
            return None

        if target_num_answers == 0:
            if len(entities_not_in_dir) < 4:
                return None
            options = random.sample(entities_not_in_dir, 4)
        elif target_num_answers is not None:
            correct_options = random.sample(entities_in_dir, target_num_answers)
            num_wrong = 4 - len(correct_options)
            if num_wrong and len(entities_not_in_dir) < num_wrong:
                return None
            wrong_options = (
                random.sample(entities_not_in_dir, num_wrong) if num_wrong else []
            )
            options = correct_options + wrong_options
            random.shuffle(options)
        else:
            max_c = min(len(entities_in_dir), 4)
            n_correct = random.randint(1, max_c)
            correct_options = random.sample(entities_in_dir, n_correct)
            num_wrong = 4 - n_correct
            if num_wrong and len(entities_not_in_dir) < num_wrong:
                return None
            wrong_options = (
                random.sample(entities_not_in_dir, num_wrong) if num_wrong else []
            )
            options = correct_options + wrong_options
            random.shuffle(options)

        type1_menu = _attach_special(
            options,
            _fifth_option(
                gold_is_none=(target_num_answers == 0),
                shuffle_none_phrase=shuffle_none_phrase,
            ),
            shuffle_special,
        )
        if len(type1_menu) != 5:
            return None
        type1_letters = ["A", "B", "C", "D", "E"]

        reasoning_lines = [
            f"**X-State**: {x_graph.format_state(active_entities)}",
            f"**Y-State**: {y_graph.format_state(active_entities)}",
        ]
        reasoning_lines.extend(
            _build_direction_reasoning(
                ref, direction, entities_in_dir,
                x_closure, y_closure, all_mentioned_entities,
            )
        )
        options_text = ", ".join(
            [f"{type1_letters[i]}. {type1_menu[i]}" for i in range(5)]
        )
        user_prompt = "Consider a map with multiple locations:\n\n"
        user_prompt += " ".join([s["text"] for s in sentences])
        user_prompt += (
            f"\n\nQuestion: Which object is in the {direction} of the {ref}? "
            f"Available options: {options_text}"
        )
        return _pack_sample(
            system_prompt, user_prompt, steps_text, reasoning_lines,
            question_type, target_num_answers,
        )

    return None


def batch_generate(
    output_file=None,
    # Type 0 — original direction question
    num_type0_1_answer=200,
    num_type0_2_answer=200,
    num_type0_undetermined=100,
    num_type0_cycle=0,
    num_type0_incomplete_pair=0,
    num_type0_none=0,
    shuffle_special=False,
    shuffle_none_phrase=False,
    # Type 1 — which-entity question
    num_type1_1_answer=200,
    num_type1_2_answer=100,
    num_type1_3_answer=0,
    num_type1_4_answer=0,
    num_type1_none=0,
    # Type 2 — count question (always 1 correct option)
    num_type2=200,
    num_type2_none=0,
    test_split=0.2,
    seed=None,
    system_prompt=None,
):
    """Generate samples in batches covering all three question types.

    After generation the samples are shuffled and split into train / test.

    Args:
        output_file: Base filename for output JSONL files.
        num_type0_1_answer: Type 0 samples with exactly 1 correct answer.
        num_type0_2_answer: Type 0 samples with exactly 2 correct answers.
        num_type0_undetermined: Type 0 samples with both axes unknown (gold E).
        num_type0_cycle: Type 0 samples with an injected both-axis cycle (gold E).
        num_type0_incomplete_pair: Type 0 one-axis with only one of the two
            remaining directions listed (gold: Cannot be determined).
        shuffle_special: mix the special option into A–E (not always E).
        shuffle_none_phrase: paraphrase "None of the Options".
        num_type1_1_answer: Type 1 samples with 1 correct entity option.
        num_type1_2_answer: Type 1 samples with 2 correct entity options.
        num_type2: Type 2 (count) samples.
        test_split: Fraction of samples reserved for the test set.
        seed: Optional RNG seed. When set, the full dataset (and its
            train/test split) is reproducible across runs and processes.
    """
    if output_file is None:
        output_file = str(
            Path(__file__).parent.parent / "data" / "spatial_sft_data_v6.jsonl"
        )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        random.seed(seed)
    train_file = output_file.replace(".jsonl", "_train.jsonl")
    test_file = output_file.replace(".jsonl", "_test.jsonl")

    all_samples = []

    # (question_type, target_num_answers, desired_count, label)
    generation_plan = [
        (0, 1, num_type0_1_answer, "Type0-1ans"),
        (0, 2, num_type0_2_answer, "Type0-2ans"),
        (0, 0, num_type0_undetermined, "Type0-undetermined-E"),
        (0, 0, num_type0_cycle, "Type0-cycle-E", {"inject_conflict": "full"}),
        (0, 0, num_type0_incomplete_pair, "Type0-incomplete-pair-E", {"incomplete_pair": True}),
        (0, 0, num_type0_none, "Type0-none-of-options", {"omit_live": True}),
        (1, 1, num_type1_1_answer, "Type1-1ans"),
        (1, 2, num_type1_2_answer, "Type1-2ans"),
        (1, 3, num_type1_3_answer, "Type1-3ans"),
        (1, 4, num_type1_4_answer, "Type1-4ans"),
        (1, 0, num_type1_none, "Type1-none-of-options"),
        (2, None, num_type2, "Type2-count"),
        (2, 0, num_type2_none, "Type2-none-of-options"),
    ]

    for item in generation_plan:
        q_type, tgt_ans, target_count, label = item[:4]
        extra = item[4] if len(item) > 4 else {}
        if not isinstance(extra, dict):
            extra = {"inject_conflict": extra} if extra else {}
        if target_count <= 0:
            continue
        generated = 0
        attempts = 0
        while generated < target_count:
            n_ent = random.randint(5, 10)
            n_sent = random.randint(n_ent, n_ent + 5)
            sample = generate_sample(
                num_entities=n_ent,
                num_sentences=n_sent,
                target_num_answers=tgt_ans,
                question_type=q_type,
                shuffle_special=shuffle_special,
                shuffle_none_phrase=shuffle_none_phrase,
                system_prompt=system_prompt,
                **extra,
            )
            attempts += 1
            if sample:
                all_samples.append(sample)
                generated += 1
                if generated % 10 == 0:
                    print(
                        f"[{label}] Generated {generated}/{target_count} "
                        f"(attempts so far: {attempts})"
                    )
        print(
            f"✅ Completed {target_count} samples for {label} "
            f"(total attempts: {attempts})"
        )

    # Shuffle & split
    num_samples = len(all_samples)
    num_test = int(num_samples * test_split)
    num_train = num_samples - num_test

    random.shuffle(all_samples)

    with open(train_file, "w", encoding="utf-8") as f:
        for sample in all_samples[:num_train]:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for sample in all_samples[num_train:]:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(
        f"✅ Done! Saved {num_train} train samples to '{train_file}' "
        f"and {num_test} test samples to '{test_file}'"
    )
    print(
        f"   Type0: {num_type0_1_answer}×1-ans + {num_type0_2_answer}×2-ans "
        f"+ {num_type0_undetermined}×undetermined-E "
        f"+ {num_type0_cycle}×cycle-E "
        f"+ {num_type0_incomplete_pair}×incomplete-pair-E "
        f"+ {num_type0_none}×none-of-options  |  "
        f"Type1: {num_type1_1_answer}×1-ans + {num_type1_2_answer}×2-ans "
        f"+ {num_type1_3_answer}×3-ans + {num_type1_4_answer}×4-ans "
        f"+ {num_type1_none}×none  |  "
        f"Type2: {num_type2}×listed + {num_type2_none}×none"
    )


# ---------------------------------------------------------------------------
# CLI (typer)
# ---------------------------------------------------------------------------
# Usage examples:
#   python generate_all_v6.py                                   # default mix → data/spatial_sft_data_v6_*.jsonl
#   python generate_all_v6.py --num-type1-1-answer 20000 \
#       --num-type0-1-answer 0 --num-type0-2-answer 0 --num-type0-undetermined 0 \
#       --num-type1-2-answer 0 --num-type2 0 --test-split 0 --seed 42
#   python generate_all_v6.py --out my_set.jsonl --seed 7
#
# Question types (numbering matches eval/clean_v6.py):
#   0 — direction, 1 — which-entity, 2 — count

app = typer.Typer(help="v6 generator: symbolic CoT, E for undetermined, no 4-ans gold.")


@app.command()
def main(
    out: str = typer.Option(
        None,
        "--out",
        help="Output JSONL base filename (writes <out>_train.jsonl / _test.jsonl). "
             "Default: <repo>/data/spatial_sft_data_all.jsonl (CWD-independent).",
    ),
    test_split: float = typer.Option(
        0.2, "--test-split", min=0.0, max=1.0,
        help="Fraction of samples reserved for the test set (0 = no split).",
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="RNG seed for a reproducible dataset.",
    ),
    # Type 0 — direction
    num_type0_1_answer: int = typer.Option(300, min=0, help="Type 0, 1 correct answer"),
    num_type0_2_answer: int = typer.Option(300, min=0, help="Type 0, 2 correct answers"),
    num_type0_undetermined: int = typer.Option(
        150, min=0,
        help="Type 0, both axes unknown (gold is E: Cannot be determined)",
    ),
    num_type0_cycle: int = typer.Option(
        0, min=0,
        help="Type 0, injected both-axis cycle (gold is E)",
    ),
    num_type0_incomplete_pair: int = typer.Option(
        0, min=0,
        help="Type 0: one axis known, only one of the two remaining "
             "directions is listed (gold: Cannot be determined)",
    ),
    num_type0_none: int = typer.Option(
        0, min=0,
        help="Type 0: live direction(s) omitted from A–D "
             "(gold: None of the Options)",
    ),
    shuffle_special: bool = typer.Option(
        False, "--shuffle-special",
        help="Mix the special option through A–E (not always letter E)",
    ),
    shuffle_none_phrase: bool = typer.Option(
        False, "--shuffle-none-phrase",
        help="Paraphrase 'None of the Options' so the wording is not fixed",
    ),
    # Type 1 — which-entity
    num_type1_1_answer: int = typer.Option(400, min=0, help="Type 1, 1 correct entity option"),
    num_type1_2_answer: int = typer.Option(200, min=0, help="Type 1, 2 correct entity options"),
    num_type1_3_answer: int = typer.Option(0, min=0, help="Type 1, 3 correct entity options"),
    num_type1_4_answer: int = typer.Option(
        0, min=0, help="Type 1, 4 correct entity options (all A–D proven)",
    ),
    num_type1_none: int = typer.Option(
        0, min=0,
        help="Type 1, no listed entity is proven (gold: None of the Options)",
    ),
    # Type 2 — count
    num_type2: int = typer.Option(750, min=0, help="Type 2 (count listed on A–D)"),
    num_type2_none: int = typer.Option(
        0, min=0,
        help="Type 2, true count omitted (gold: None of the Options)",
    ),
    system_prompt_file: str | None = typer.Option(
        None,
        "--system-prompt-file",
        help="Replace the default SFT system prompt with this file. "
             "Default (omit) is the built-in v6 prompt.",
    ),
) -> None:
    """Generate the dataset. Defaults reproduce the 2100-sample training mix."""
    if out is None:
        out = str(Path(__file__).parent.parent / "data" / "spatial_sft_data_v6.jsonl")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    batch_generate(
        out,
        num_type0_1_answer=num_type0_1_answer,
        num_type0_2_answer=num_type0_2_answer,
        num_type0_undetermined=num_type0_undetermined,
        num_type0_cycle=num_type0_cycle,
        num_type0_incomplete_pair=num_type0_incomplete_pair,
        num_type0_none=num_type0_none,
        shuffle_special=shuffle_special,
        shuffle_none_phrase=shuffle_none_phrase,
        num_type1_1_answer=num_type1_1_answer,
        num_type1_2_answer=num_type1_2_answer,
        num_type1_3_answer=num_type1_3_answer,
        num_type1_4_answer=num_type1_4_answer,
        num_type1_none=num_type1_none,
        num_type2=num_type2,
        num_type2_none=num_type2_none,
        test_split=test_split,
        seed=seed,
        system_prompt=(
            Path(system_prompt_file).read_text(encoding="utf-8").strip()
            if system_prompt_file
            else None
        ),
    )


if __name__ == "__main__":
    app()
