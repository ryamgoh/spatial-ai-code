import random
import json
import itertools
from collections import defaultdict
from pathlib import Path

import typer

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
NONE_LABEL = "None of these is proven"


def _format_options(options: list, letters: list[str], include_e: bool) -> str:
    parts = [f"{letters[i]}. {options[i]}" for i in range(len(options))]
    if include_e:
        parts.append(f"E. {NONE_LABEL}")
    return ", ".join(parts)


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

        is_east = (ref, entity) in x_closure
        is_west = (entity, ref) in x_closure
        is_north = (ref, entity) in y_closure
        is_south = (entity, ref) in y_closure

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

def generate_sample(num_entities=5, num_sentences=6, target_num_answers=None,
                    question_type=0, include_option_e=False, none_of_above=False):
    """Generate a single SFT training sample.

    Args:
        num_entities: Number of entities to use.
        num_sentences: Number of relation sentences to generate.
        target_num_answers: Desired number of correct answers.
            Type 0 — 1, 2, or 4 (direction ambiguity).
            Type 1 — number of correct entity options among A-D (1, 2, or 3).
            Type 2 — ignored (count question, always exactly 1 correct count).
            If None, any number of answers is accepted.
        question_type:
            0 — original direction question
            1 — which-entity question ("Which object …")
            2 — count question  ("How many objects …")
            Numbering matches eval/clean_v5.py.

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

    # Shared system prompt
    system_prompt = (
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
        "CRITICAL RULE FOR FINAL DEDUCTION:\nIf the relationship on a "
        "specific axis (X or Y) cannot be determined from the given "
        "information, you MUST acknowledge the uncertainty. Combine the known "
        "and unknown axes to list ALL possible directions (e.g., \"Northeast "
        "or Southeast\", or all 4 directions if completely unknown). Do not "
        "guess a single direction if the evidence is insufficient."
    )

    option_letters = ["A", "B", "C", "D"]

    # ------------------------------------------------------------------
    # TYPE 0 — Original direction question
    # ------------------------------------------------------------------
    if question_type == 0:
        if not mentioned_pairs:
            return None

        if target_num_answers is not None:
            candidate_pairs = []
            for a, b in mentioned_pairs:
                x_determined = (a, b) in x_closure or (b, a) in x_closure
                y_determined = (a, b) in y_closure or (b, a) in y_closure

                if x_determined and y_determined:
                    num_answers = 1
                elif x_determined or y_determined:
                    num_answers = 2
                else:
                    num_answers = 4

                if num_answers == target_num_answers:
                    candidate_pairs.append((a, b))

            if not candidate_pairs:
                return None
            target, ref = random.choice(candidate_pairs)
        else:
            target, ref = random.choice(mentioned_pairs)

        # X-axis
        if (ref, target) in x_closure:
            ans_x_list = ["East"]
            x_reasoning = f"The {target} is strictly to the East of the {ref}."
        elif (target, ref) in x_closure:
            ans_x_list = ["West"]
            x_reasoning = f"The {target} is strictly to the West of the {ref}."
        else:
            ans_x_list = ["East", "West"]
            x_reasoning = (
                f"The exact X-axis relationship between the {target} and the "
                f"{ref} cannot be determined (could be East or West)."
            )

        # Y-axis
        if (ref, target) in y_closure:
            ans_y_list = ["North"]
            y_reasoning = f"The {target} is strictly to the North of the {ref}."
        elif (target, ref) in y_closure:
            ans_y_list = ["South"]
            y_reasoning = f"The {target} is strictly to the South of the {ref}."
        else:
            ans_y_list = ["North", "South"]
            y_reasoning = (
                f"The exact Y-axis relationship between the {target} and the "
                f"{ref} cannot be determined (could be North or South)."
            )

        possible_dirs = []
        for y in ans_y_list:
            for x in ans_x_list:
                possible_dirs.append(f"{y}{x.lower()}")

        all_dir_options = [
            "North", "South", "East", "West",
            "Northeast", "Northwest", "Southeast", "Southwest",
        ]
        if none_of_above:
            wrong = [d for d in all_dir_options if d not in possible_dirs]
            if len(wrong) < 4:
                return None
            options = random.sample(wrong, 4)
            random.shuffle(options)
            correct_letters = []
            answer_str = "E"
            include_option_e = True
        else:
            options = list(possible_dirs)
            while len(options) < 4:
                d = random.choice(all_dir_options)
                if d not in options:
                    options.append(d)
            random.shuffle(options)
            correct_letters = [
                option_letters[i] for i in range(len(options))
                if options[i] in possible_dirs
            ]
            answer_str = ", ".join(correct_letters)

        if len(possible_dirs) == 1:
            target_dir_str = possible_dirs[0]
            conclusion = (
                f"The {target} is to the {target_dir_str} of the {ref}."
            )
        elif len(possible_dirs) == 2:
            target_dir_str = f"{possible_dirs[0]} or {possible_dirs[1]}"
            conclusion = (
                f"Due to partial information, the {target} could be to the "
                f"{target_dir_str} of the {ref}."
            )
        else:
            target_dir_str = "Northeast, Northwest, Southeast, or Southwest"
            conclusion = (
                f"The spatial relationship is completely undetermined. The "
                f"{target} could be to the {target_dir_str} of the {ref}."
            )
        if none_of_above:
            conclusion = (
                f"None of options A–D is a proven direction "
                f"(possible: {', '.join(possible_dirs)}). The answer is E."
            )

        thinking_content = f"{steps_text[0]}\n\n"
        for step in steps_text[1:]:
            thinking_content += f"{step}\n\n"
        thinking_content += "### Final Deduction\n"
        thinking_content += f"**Target**: {target}\n"
        thinking_content += f"**Reference**: {ref}\n"
        thinking_content += f"**X-Axis Analysis**: {x_reasoning}\n"
        thinking_content += f"**Y-Axis Analysis**: {y_reasoning}\n"
        thinking_content += f"**Conclusion**: {conclusion}\n"

        final_deduction = (
            f"<think>\n{thinking_content}\n</think>\nAnswer: {answer_str}"
        )

        options_text = _format_options(options, option_letters, include_option_e)
        user_prompt = "Consider a map with multiple locations:\n\n"
        user_prompt += " ".join([s["text"] for s in sentences])
        user_prompt += (
            f"\n\nQuestion: In which direction is the {target} relative to "
            f"the {ref}? Available options: {options_text}"
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

        # ---- Reasoning ----
        reasoning_lines = _build_direction_reasoning(
            ref, direction, entities_in_dir,
            x_closure, y_closure, all_mentioned_entities,
        )
        reasoning_lines.append(f"**Count**: {correct_count}")
        if correct_count == 0:
            reasoning_lines.append(
                f"**Conclusion**: There are no objects in the {direction} of "
                f"the {ref}."
            )
        elif correct_count == 1:
            reasoning_lines.append(
                f"**Conclusion**: There is 1 object in the {direction} of "
                f"the {ref}."
            )
        else:
            reasoning_lines.append(
                f"**Conclusion**: There are {correct_count} objects in the "
                f"{direction} of the {ref}."
            )

        thinking_content = f"{steps_text[0]}\n\n"
        for step in steps_text[1:]:
            thinking_content += f"{step}\n\n"
        thinking_content += "### Final Deduction\n"
        thinking_content += "\n".join(reasoning_lines) + "\n"

        # ---- Options (4 distinct integers) ----
        max_count = len(all_mentioned_entities) - 1
        option_values = [correct_count]
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

        if none_of_above:
            option_values = [i for i in range(0, max_count + 2) if i != correct_count][:4]
            if len(option_values) < 4:
                return None
            random.shuffle(option_values)
            answer_str = "E"
            include_option_e = True
            reasoning_lines.append(
                f"**Conclusion**: The count {correct_count} is not among "
                f"options A–D, so none of them is proven. The answer is E."
            )
            thinking_content = f"{steps_text[0]}\n\n"
            for step in steps_text[1:]:
                thinking_content += f"{step}\n\n"
            thinking_content += "### Final Deduction\n"
            thinking_content += "\n".join(reasoning_lines) + "\n"
        else:
            answer_str = ", ".join(
                option_letters[i] for i in range(4)
                if option_values[i] == correct_count
            )

        final_deduction = (
            f"<think>\n{thinking_content}\n</think>\nAnswer: {answer_str}"
        )

        options_text = _format_options(
            [str(v) for v in option_values], option_letters, include_option_e
        )
        user_prompt = "Consider a map with multiple locations:\n\n"
        user_prompt += " ".join([s["text"] for s in sentences])
        user_prompt += (
            f"\n\nQuestion: How many objects are in the {direction} of the "
            f"{ref}? Available options: {options_text}"
        )

    # ------------------------------------------------------------------
    # TYPE 1 — Which-entity question
    # ------------------------------------------------------------------
    elif question_type == 1:
        min_correct = target_num_answers if target_num_answers is not None else 1

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
                need_wrong = 4 if none_of_above else (4 - min_correct)
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

        if none_of_above:
            if len(entities_not_in_dir) < 4:
                return None
            options = random.sample(entities_not_in_dir, 4)
            random.shuffle(options)
            correct_options = []
            correct_letters = []
            answer_str = "E"
            include_option_e = True
        else:
            # Select exactly the requested number of correct entities for options
            if target_num_answers is not None:
                correct_options = random.sample(entities_in_dir, target_num_answers)
            else:
                max_c = min(len(entities_in_dir), 3)
                n_correct = random.randint(1, max_c)
                correct_options = random.sample(entities_in_dir, n_correct)

            num_wrong = 4 - len(correct_options)
            wrong_options = random.sample(entities_not_in_dir, num_wrong)

            options = correct_options + wrong_options
            random.shuffle(options)

            correct_set = set(correct_options)
            correct_letters = [
                option_letters[i] for i in range(4) if options[i] in correct_set
            ]
            answer_str = ", ".join(correct_letters)

        # ---- Reasoning ----
        reasoning_lines = _build_direction_reasoning(
            ref, direction, entities_in_dir,
            x_closure, y_closure, all_mentioned_entities,
        )

        # Conclusion mentioning which option entities are correct
        if len(correct_options) == 1:
            reasoning_lines.append(
                f"**Conclusion**: Among the given options, the "
                f"{correct_options[0]} is in the {direction} of the {ref}."
            )
        elif len(correct_options) == 2:
            reasoning_lines.append(
                f"**Conclusion**: Among the given options, the "
                f"{correct_options[0]} and the {correct_options[1]} are in "
                f"the {direction} of the {ref}."
            )
        else:
            parts = [f"the {e}" for e in correct_options]
            ent_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"
            reasoning_lines.append(
                f"**Conclusion**: Among the given options, {ent_str} are in "
                f"the {direction} of the {ref}."
            )
        if none_of_above:
            reasoning_lines.append(
                f"**Conclusion**: None of options A–D is proven to lie in "
                f"the {direction} of the {ref}. The answer is E."
            )

        thinking_content = f"{steps_text[0]}\n\n"
        for step in steps_text[1:]:
            thinking_content += f"{step}\n\n"
        thinking_content += "### Final Deduction\n"
        thinking_content += "\n".join(reasoning_lines) + "\n"

        final_deduction = (
            f"<think>\n{thinking_content}\n</think>\nAnswer: {answer_str}"
        )

        options_text = _format_options(options, option_letters, include_option_e)
        user_prompt = "Consider a map with multiple locations:\n\n"
        user_prompt += " ".join([s["text"] for s in sentences])
        user_prompt += (
            f"\n\nQuestion: Which object is in the {direction} of the {ref}? "
            f"Available options: {options_text}"
        )

    else:
        return None

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": final_deduction},
        ]
    }


def batch_generate(
    output_file=None,
    # Type 0 — original direction question
    num_type0_1_answer=200,
    num_type0_2_answer=200,
    num_type0_4_answer=100,
    # Type 1 — which-entity question
    num_type1_1_answer=200,
    num_type1_2_answer=100,
    num_type1_4_answer=0,
    # Type 2 — count question (always 1 correct option)
    num_type2=200,
    include_option_e=False,
    num_none_dir=0,
    num_none_which=0,
    num_none_count=0,
    test_split=0.2,
    seed=None,
):
    """Generate samples in batches covering all three question types.

    After generation the samples are shuffled and split into train / test.

    Args:
        output_file: Base filename for output JSONL files.
        num_type0_1_answer: Type 0 samples with exactly 1 correct answer.
        num_type0_2_answer: Type 0 samples with exactly 2 correct answers.
        num_type0_4_answer: Type 0 samples with exactly 4 correct answers.
        num_type1_1_answer: Type 1 samples with 1 correct entity option.
        num_type1_2_answer: Type 1 samples with 2 correct entity options.
        num_type1_4_answer: Type 1 samples with 4 correct entity options
            (all four A–D gold; matches TQA-Corr which-4-ans).
        num_type2: Type 2 (count) samples.
        include_option_e: Append ``E. None of these is proven`` to every item.
        num_none_dir: Type 0 items whose A–D dirs are all wrong (gold E).
        num_none_which: Type 1 items whose A–D entities are all wrong (gold E).
        num_none_count: Type 2 items whose A–D counts are all wrong (gold E).
        test_split: Fraction of samples reserved for the test set.
        seed: Optional RNG seed. When set, the full dataset (and its
            train/test split) is reproducible across runs and processes.
    """
    if output_file is None:
        output_file = str(Path(__file__).parent.parent / "data" / "spatial_sft_data_uncertainty.jsonl")
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
        (0, 4, num_type0_4_answer, "Type0-4ans"),
        (1, 1, num_type1_1_answer, "Type1-1ans"),
        (1, 2, num_type1_2_answer, "Type1-2ans"),
        (1, 4, num_type1_4_answer, "Type1-4ans"),
        (2, None, num_type2, "Type2-count"),
    ]

    for q_type, tgt_ans, target_count, label in generation_plan:
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
                include_option_e=include_option_e,
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

    none_plan = [
        (0, num_none_dir, "None-dir"),
        (1, num_none_which, "None-which"),
        (2, num_none_count, "None-count"),
    ]
    for q_type, target_count, label in none_plan:
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
                target_num_answers=1 if q_type != 2 else None,
                question_type=q_type,
                include_option_e=True,
                none_of_above=True,
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
            if attempts > max(5000, target_count * 200) and generated == 0:
                raise RuntimeError(f"{label}: no samples after {attempts} attempts")
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
        f"+ {num_type0_4_answer}×4-ans  |  "
        f"Type1: {num_type1_1_answer}×1-ans + {num_type1_2_answer}×2-ans  |  "
        f"Type2: {num_type2}"
    )


# ---------------------------------------------------------------------------
# CLI (typer)
# ---------------------------------------------------------------------------
# Usage examples:
#   python generate_all.py                                   # default 2100-sample mix
#   python generate_all.py --num-type1-1-answer 20000 \
#       --num-type0-1-answer 0 --num-type0-2-answer 0 --num-type0-4-answer 0 \
#       --num-type1-2-answer 0 --num-type2 0 --test-split 0 --seed 42
#   python generate_all.py --out my_set.jsonl --seed 7
#
# Question types (numbering matches eval/clean_v5.py):
#   0 — direction, 1 — which-entity, 2 — count

app = typer.Typer(help="Generate synthetic SFT data for spatial reasoning.")


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
    num_type0_4_answer: int = typer.Option(150, min=0, help="Type 0, 4 correct answers"),
    # Type 1 — which-entity
    num_type1_1_answer: int = typer.Option(400, min=0, help="Type 1, 1 correct entity option"),
    num_type1_2_answer: int = typer.Option(200, min=0, help="Type 1, 2 correct entity options"),
    num_type1_4_answer: int = typer.Option(0, min=0, help="Type 1, 4 correct entity options"),
    # Type 2 — count
    num_type2: int = typer.Option(750, min=0, help="Type 2 (count) samples"),
    include_option_e: bool = typer.Option(
        False, "--include-option-e", help="Append E. None of these is proven."
    ),
    num_none_dir: int = typer.Option(0, min=0, help="Type 0 gold-E (wrong A–D dirs)"),
    num_none_which: int = typer.Option(0, min=0, help="Type 1 gold-E (wrong A–D entities)"),
    num_none_count: int = typer.Option(0, min=0, help="Type 2 gold-E (wrong A–D counts)"),
) -> None:
    """Generate the dataset. Defaults reproduce the 2100-sample training mix."""
    if out is None:
        out = str(Path(__file__).parent.parent / "data" / "spatial_sft_data_all.jsonl")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    batch_generate(
        out,
        num_type0_1_answer=num_type0_1_answer,
        num_type0_2_answer=num_type0_2_answer,
        num_type0_4_answer=num_type0_4_answer,
        num_type1_1_answer=num_type1_1_answer,
        num_type1_2_answer=num_type1_2_answer,
        num_type1_4_answer=num_type1_4_answer,
        num_type2=num_type2,
        include_option_e=include_option_e,
        num_none_dir=num_none_dir,
        num_none_which=num_none_which,
        num_none_count=num_none_count,
        test_split=test_split,
        seed=seed,
    )


if __name__ == "__main__":
    app()
