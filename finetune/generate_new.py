import random
import json
import itertools
from collections import defaultdict

# Preset entity name library (can be extended for specific vertical domains)
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


def generate_sample(num_entities=5, num_sentences=6):
    """Generate a single SFT training sample"""
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

    # ================= Phase 3: Generate QA pairs that support multiple possible answers =================
    x_closure = x_graph.get_transitive_closure()
    y_closure = y_graph.get_transitive_closure()

    # Randomly choose a pair of entities as the question target (no longer requiring full connectivity)
    target, ref = random.choice(pairs)

    # 1. Independently determine the X-axis relation
    if (ref, target) in x_closure:
        ans_x_list = ["East"]
        x_reasoning = f"The {target} is strictly to the East of the {ref}."
    elif (target, ref) in x_closure:
        ans_x_list = ["West"]
        x_reasoning = f"The {target} is strictly to the West of the {ref}."
    else:
        ans_x_list = ["East", "West"]  # Unknown relation: both are possible
        x_reasoning = f"The exact X-axis relationship between the {target} and the {ref} cannot be determined (could be East or West)."

    # 2. Independently determine the Y-axis relation
    if (ref, target) in y_closure:
        ans_y_list = ["North"]
        y_reasoning = f"The {target} is strictly to the North of the {ref}."
    elif (target, ref) in y_closure:
        ans_y_list = ["South"]
        y_reasoning = f"The {target} is strictly to the South of the {ref}."
    else:
        ans_y_list = ["North", "South"]  # Unknown relation: both are possible
        y_reasoning = f"The exact Y-axis relationship between the {target} and the {ref} cannot be determined (could be North or South)."

    # 3. Use the Cartesian product to enumerate all possible 2D directions
    possible_dirs = []
    for y in ans_y_list:
        for x in ans_x_list:
            possible_dirs.append(f"{y}{x.lower()}")

    # 4. Generate multiple choice options
    all_directions = [
        "North",
        "South",
        "East",
        "West",
        "Northeast",
        "Northwest",
        "Southeast",
        "Southwest",
    ]
    options = list(possible_dirs)
    while len(options) < 4:
        d = random.choice(all_directions)
        if d not in options:
            options.append(d)
    random.shuffle(options)
    option_letters = ["A", "B", "C", "D"]
    options_with_letters = {option_letters[i]: options[i] for i in range(len(options))}
    correct_letters = [
        option_letters[i] for i in range(len(options)) if options[i] in possible_dirs
    ]
    answer_str = ", ".join(correct_letters)

    # 5. Generate different conclusion wording based on the number of possibilities
    if len(possible_dirs) == 1:
        target_dir_str = possible_dirs[0]
        conclusion = f"The {target} is to the {target_dir_str} of the {ref}."
    elif len(possible_dirs) == 2:
        target_dir_str = f"{possible_dirs[0]} or {possible_dirs[1]}"
        conclusion = f"Due to partial information, the {target} could be to the {target_dir_str} of the {ref}."
    else:
        target_dir_str = "Northeast, Northwest, Southeast, or Southwest"
        conclusion = f"The spatial relationship is completely undetermined. The {target} could be to the {target_dir_str} of the {ref}."

    # Build the final deduction with thinking tags
    thinking_content = f"{steps_text[0]}\n\n"
    for i, step in enumerate(steps_text[1:], 1):
        thinking_content += f"{step}\n\n"
    thinking_content += f"### Final Deduction\n"
    thinking_content += f"**Target**: {target}\n"
    thinking_content += f"**Reference**: {ref}\n"
    thinking_content += f"**X-Axis Analysis**: {x_reasoning}\n"
    thinking_content += f"**Y-Axis Analysis**: {y_reasoning}\n"
    thinking_content += f"**Conclusion**: {conclusion}\n"

    final_deduction = f"<think>\n{thinking_content}\n</think>\nAnswer: {answer_str}"
    # =================================================================

    # Update the system prompt to inform the model that multiple valid answers are allowed
    system_prompt = """You are an advanced spatial reasoning agent. Process the spatial relations step-by-step.
First, initialize by detecting all entities in the text to establish the global scope. 
Then, step-by-step, extract relations and update the spatial state for the X-axis (West to East) and Y-axis (South to North), tracking ONLY the entities that have been mentioned so far.
Rules for State Representation:
1. Use "<" for strict ordering (e.g., A < B means A is West/South of B).
2. Group topologically equivalent entities using "{}" (e.g., {A, B} < C).
3. Merge relations into maximal chains to represent the spatial topology clearly.
4. Keep isolated active entities separated by commas until they are connected.

CRITICAL RULE FOR FINAL DEDUCTION:
If the relationship on a specific axis (X or Y) cannot be determined from the given information, you MUST acknowledge the uncertainty. Combine the known and unknown axes to list ALL possible directions (e.g., "Northeast or Southeast", or all 4 directions if completely unknown). Do not guess a single direction if the evidence is insufficient."""

    options_text = ", ".join(
        [f"{option_letters[i]}. {options[i]}" for i in range(len(options))]
    )
    user_prompt = "Consider a map with multiple locations:\n\n"
    user_prompt += " ".join([s["text"] for s in sentences])
    user_prompt += f"\n\nQuestion: In which direction is the {target} relative to the {ref}? Available options: {options_text}"

    assistant_prompt = final_deduction

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prompt},
        ]
    }


def batch_generate(output_file="spatial_sft_data_uncertainty.jsonl", num_samples=100):
    """Generate samples in batches and save them in JSONL format"""
    valid_samples = 0
    with open(output_file, "w", encoding="utf-8") as f:
        while valid_samples < num_samples:
            n_ent = random.randint(4, 7)
            n_sent = random.randint(
                n_ent, n_ent + 2
            )  # Slightly reduce the number of sentences to increase the chance of "multiple-answer" cases

            sample = generate_sample(num_entities=n_ent, num_sentences=n_sent)
            if sample:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                valid_samples += 1
                if valid_samples % 10 == 0:
                    print(f"Generated {valid_samples}/{num_samples} samples...")

    print(
        f"✅ Done! Successfully saved {num_samples} high-quality SFT samples to '{output_file}'"
    )


if __name__ == "__main__":
    batch_generate("spatial_sft_data_uncertainty.jsonl", 500)
