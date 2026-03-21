import random
import re
import os
import sys


def process_docs_local_train(dataset):
    """
    Convert local training data with messages format to lm-eval format.

    Input format (your data):
    {
      "messages": [
       {"role": "system", "content": "..."},
        {"role": "user", "content": "Question..."},
        {"role": "assistant", "content": "Thinking: ...Answer: A"}
      ]
    }

    Output format (lm-eval):
    {
      "text": "user content",
      "oracle_option": "A"
    }
    """

    def convert(doc):
        user_content = ""
        oracle_options = []

        for msg in doc["messages"]:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                match = re.search(r"Answer:\s*([A-D](?:,\s*[A-D])*)", msg["content"])
                if match:
                    options_str = match.group(1)
                    oracle_options = [opt.strip() for opt in options_str.split(",")]

        return {
            "text": user_content,
            # Join with comma (no spaces) for consistent parsing: "A,D"
            "oracle_option": ",".join(oracle_options) if oracle_options else "",
        }

    return dataset.map(convert)


def process_docs(dataset, seed=42):
    """
    Shuffle answer choices to eliminate position bias.
    The correct answer is randomly placed among the 4 positions.
    """
    random.seed(seed)

    def shuffle_choices(doc):
        choices = [
            doc["distractor1"],
            doc["distractor2"],
            doc["distractor3"],
            doc["correct_answer"],
        ]
        correct_answer = doc["correct_answer"]

        random.shuffle(choices)

        correct_index = choices.index(correct_answer)

        return {
            **doc,
            "choice_a": choices[0],
            "choice_b": choices[1],
            "choice_c": choices[2],
            "choice_d": choices[3],
            "correct_index": correct_index,
        }

    return dataset.map(shuffle_choices)

def _normalize(text: str) -> str:
    """Normalize curly/smart quotes to ASCII apostrophe so object names
    parsed from the map description always match those parsed from options."""
    return (
        text
        .replace("\u2018", "'").replace("\u2019", "'")   # '' → '
        .replace("\u201c", '"').replace("\u201d", '"')   # "" → "
    )


# ──────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────

def parse_problem(text: str):
    """
    Returns (objects: list[str], relations: list[(a, dir, b)], question_part: str).
    'direction' values are lowercase: 'northeast', 'southeast', 'southwest', 'northwest'.
    """
    text = _normalize(text)
    split_marker = "Please answer"
    if split_marker in text:
        idx = text.index(split_marker)
        map_part = text[:idx]
        question_part = text[idx:]
    else:
        map_part = text
        question_part = ""

    objects: set[str] = set()

    # First object declared with "X is in the map."
    m = re.search(r"([^.]+?) is in the map", map_part)
    if m:
        objects.add(m.group(1).strip())

    # All directional relations: "X is to the [Dir] of Y."
    rel_re = re.compile(
        r"([^.]+?) is to the (Northeast|Northwest|Southeast|Southwest) of ([^.]+?)\.",
        re.IGNORECASE,
    )
    relations = []
    for m in rel_re.finditer(map_part):
        a = m.group(1).strip()
        d = m.group(2).strip().lower()
        b = m.group(3).strip()
        relations.append((a, d, b))
        objects.add(a)
        objects.add(b)

    return list(objects), relations, question_part


def detect_type(question_part: str) -> int:
    if "In which direction is" in question_part:
        return 0
    if "Which object is in the" in question_part:
        return 1
    if "How many objects are in the" in question_part:
        return 2
    return -1


def parse_options(question_part: str) -> dict[str, str]:
    """Parse 'A. Value' lines → {'A': 'Value', ...}."""
    opts = {}
    for m in re.finditer(r"\b([A-D])\.\s*([^\n]+)", question_part):
        key = m.group(1)
        val = m.group(2).strip().rstrip(".")
        opts[key] = val
    return opts


# ──────────────────────────────────────────────────────────
# Constraint graphs & transitive closure
# ──────────────────────────────────────────────────────────

def transitive_closure(objects: list[str], edges: list[tuple]) -> dict[str, set]:
    """
    reach[a] = set of b such that a > b is provably derivable.
    Uses iterative fixed-point expansion.
    """
    reach: dict[str, set] = {obj: set() for obj in objects}
    for (a, b) in edges:
        if a in reach:
            reach[a].add(b)

    changed = True
    while changed:
        changed = False
        for a in objects:
            before = len(reach[a])
            extras: set = set()
            for b in reach[a]:
                extras |= reach.get(b, set())
            reach[a] |= extras
            if len(reach[a]) > before:
                changed = True
    return reach


def build_order_graphs(objects: list[str], relations: list[tuple]):
    """
    Returns (x_reach, y_reach).
      x_reach[a] contains b  →  x_a > x_b  (a is east  of b, provably)
      y_reach[a] contains b  →  y_a > y_b  (a is north of b, provably)
    """
    x_edges, y_edges = [], []

    for (a, d, b) in relations:
        if d == "northeast":      # x_a > x_b,  y_a > y_b
            x_edges.append((a, b))
            y_edges.append((a, b))
        elif d == "southeast":    # x_a > x_b,  y_a < y_b  →  y_b > y_a
            x_edges.append((a, b))
            y_edges.append((b, a))
        elif d == "southwest":    # x_a < x_b  →  x_b > x_a,  y_a < y_b  →  y_b > y_a
            x_edges.append((b, a))
            y_edges.append((b, a))
        elif d == "northwest":    # x_a < x_b  →  x_b > x_a,  y_a > y_b
            x_edges.append((b, a))
            y_edges.append((a, b))

    return (
        transitive_closure(objects, x_edges),
        transitive_closure(objects, y_edges),
    )


def get_rel(reach: dict, a: str, b: str) -> str:
    """
    Relation of a vs b on one axis.
      'gt'      → a > b  (proved)
      'lt'      → a < b  (proved, because b > a)
      'unknown' → cannot determine
    """
    if b in reach.get(a, set()):
        return "gt"
    if a in reach.get(b, set()):
        return "lt"
    return "unknown"


# ──────────────────────────────────────────────────────────
# Direction checks
# ──────────────────────────────────────────────────────────

def direction_possible(x_rel: str, y_rel: str, direction: str) -> bool:
    """
    Is it POSSIBLE for A to be in `direction` of B,
    given x_rel / y_rel are A's relation to B?
    'gt' on x means A is east of B, etc.
    """
    d = direction.strip().lower()
    # Broad single-axis directions
    if d == "north":
        return y_rel != "lt"
    if d == "south":
        return y_rel != "gt"
    if d == "east":
        return x_rel != "lt"
    if d == "west":
        return x_rel != "gt"
    # Diagonal directions
    needs_east  = d in ("northeast", "southeast")
    needs_north = d in ("northeast", "northwest")
    x_ok = (needs_east  and x_rel != "lt") or (not needs_east  and x_rel != "gt")
    y_ok = (needs_north and y_rel != "lt") or (not needs_north and y_rel != "gt")
    return x_ok and y_ok


def direction_definite(x_rel: str, y_rel: str, direction: str) -> bool:
    """
    Is it DEFINITELY TRUE that A is in `direction` of B?
    Used for Type-2 counting.
    """
    d = direction.strip().lower()
    if d == "north":
        return y_rel == "gt"
    if d == "south":
        return y_rel == "lt"
    if d == "east":
        return x_rel == "gt"
    if d == "west":
        return x_rel == "lt"
    if d == "northeast":
        return x_rel == "gt" and y_rel == "gt"
    if d == "southeast":
        return x_rel == "gt" and y_rel == "lt"
    if d == "southwest":
        return x_rel == "lt" and y_rel == "lt"
    if d == "northwest":
        return x_rel == "lt" and y_rel == "gt"
    return False


# ──────────────────────────────────────────────────────────
# Main solver
# ──────────────────────────────────────────────────────────

def solve(text: str) -> str:
    """
    Parse and solve one spatial question.
    Returns comma-separated valid option letters, e.g. "A", "A,C", "A,B,C,D".
    """
    objects, relations, question_part = parse_problem(text)
    if not question_part:
        return "Error: no question found"

    q_type = detect_type(question_part)
    options = parse_options(question_part)
    if not options:
        return "Error: no options found"

    x_reach, y_reach = build_order_graphs(objects, relations)
    valid: list[str] = []

    # ── Type 0: "In which direction is X relative to Y?" ──────────────────
    if q_type == 0:
        m = re.search(
            r"In which direction is ([^?]+?) relative to ([^?]+?)\?", question_part
        )
        if not m:
            return "Error: cannot parse type-0 question"
        obj_x = m.group(1).strip()
        obj_y = m.group(2).strip()
        x_rel = get_rel(x_reach, obj_x, obj_y)
        y_rel = get_rel(y_reach, obj_x, obj_y)
        for key in sorted(options):
            if direction_possible(x_rel, y_rel, options[key]):
                valid.append(key)

    # ── Type 1: "Which object is in the [Dir] of X?" ──────────────────────
    elif q_type == 1:
        m = re.search(
            r"Which object is in the (\w+) of ([^?]+?)\?", question_part
        )
        if not m:
            return "Error: cannot parse type-1 question"
        direction = m.group(1).strip()
        obj_ref = m.group(2).strip()
        for key in sorted(options):
            candidate = options[key]
            x_rel = get_rel(x_reach, candidate, obj_ref)
            y_rel = get_rel(y_reach, candidate, obj_ref)
            if direction_possible(x_rel, y_rel, direction):
                valid.append(key)

    # ── Type 2: "How many objects are in the [Dir] of X?" ─────────────────
    elif q_type == 2:
        m = re.search(
            r"How many objects are in the (\w+) of ([^?]+?)\?", question_part
        )
        if not m:
            return "Error: cannot parse type-2 question"
        direction = m.group(1).strip()
        obj_ref = m.group(2).strip()

        count = 0
        for obj in objects:
            if obj == obj_ref:
                continue
            x_rel = get_rel(x_reach, obj, obj_ref)
            y_rel = get_rel(y_reach, obj, obj_ref)
            if direction_definite(x_rel, y_rel, direction):
                count += 1

        for key in sorted(options):
            try:
                if int(options[key]) == count:
                    valid.append(key)
            except ValueError:
                pass

    else:
        return "Error: unknown question type"

    return ",".join(valid) if valid else "No valid options found"


def filter_spatialmap_and_update_oracle_answer_new(dataset):

    dataset = dataset.filter(lambda x: bool(re.match(r"^spatialmap\.", x["id"])))

    def add_oracle(doc):
        doc["oracle_option"] = solve(doc["text"])
        return doc

    return dataset.map(add_oracle)


def filter_spatialmap(dataset):
    """Filter dataset to only include rows where id starts with 'spatialmap.'"""
    return dataset.filter(lambda x: bool(re.match(r"^spatialmap\.", x["id"])))


def filter_spatialmap_first_type(dataset):
    """Filter dataset to only include rows where id matches 'spatialmap.tqa.[number].1'."""
    return dataset.filter(
        lambda x: bool(re.match(r"^spatialmap\.tqa\.\d+\.1$", x["id"]))
    )


def filter_spatialmap_zero_type(dataset):
    """Filter dataset to only include rows where id matches 'spatialmap.tqa.[number].0'."""
    return dataset.filter(
        lambda x: bool(re.match(r"^spatialmap\.tqa\.\d+\.0$", x["id"]))
    )


def process_docs_with_rag(dataset):
    """Process docs with RAG augmentation."""

    from rag import RAGManager

    # RAG Config
    context_k = 3
    context_template = "- {text}"
    context_separator = "\n"
    query_field = "text"
    context_field = "context"
    corpus_paths = "../spatial_knowledge.docx"
    chunk_size = 800
    chunk_overlap = 100
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    # Filter to spatialmap
    dataset = dataset.filter(lambda x: bool(re.match(r"^spatialmap\.", x["id"])))

    rag_manager = RAGManager()
    retriever = rag_manager.get_retriever(
        name="spatial_knowledge",
        corpus_paths=[corpus_paths],
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    def add_rag(doc):
        query = doc.get(query_field, "")
        context = retriever.get_context(
            query=query,
            k=context_k,
            template=context_template,
            separator=context_separator,
        )
        doc[context_field] = context
        return doc

    return dataset.map(add_rag)


def macro_f1(items):
    """Compute macro F1 score for multiclass classification."""
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return f1_score(golds, preds, average="macro")


def mcc(items):
    """Compute Matthews Correlation Coefficient for multiclass classification."""
    from sklearn.metrics import matthews_corrcoef

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return matthews_corrcoef(golds, preds)


def extract_choice(response):
    """
    Extract the answer choice (A, B, C, D) from a generative response.
    Handles various formats like:
    - "A" or "B." or "C)" or "(D)"
    - "The answer is A"
    - "A. nitrogen hormones"
    """
    if not response:
        return -1

    response = response.strip()

    # Look for standalone letter at start
    match = re.match(r"^\s*[(\[]?\s*([A-Da-d])[)\].:)]?\s", response)
    if match:
        return ord(match.group(1).upper()) - ord("A")

    # Look for "answer is X" pattern
    match = re.search(
        r"(?:answer|choice|option)\s+(?:is\s+)?[(\[]?\s*([A-Da-d])[)\].:)]?",
        response,
        re.IGNORECASE,
    )
    if match:
        return ord(match.group(1).upper()) - ord("A")

    # Look for any A/B/C/D with word boundary
    match = re.search(r"\b([A-Da-d])\b", response)
    if match:
        return ord(match.group(1).upper()) - ord("A")

    return -1


def process_gen_response(items):
    """
    Process generative responses for multiple choice questions.
    Items is a list of (response, correct_index) tuples.
    Returns accuracy.
    """
    correct = 0
    total = len(items)

    for response, correct_index in items:
        predicted = extract_choice(response)
        if predicted == correct_index:
            correct += 1

    return correct / total if total > 0 else 0.0


def acc_gen(items):
    """Accuracy metric for generative multiple choice.
    items is [gold, filtered_resps] where:
    - gold: str like "A", "B", "C", "D"
    - filtered_resps: list like ["D"] (from filter)
    """
    # items is [gold, filtered_resps] - unpack it
    gold = items[0]
    filtered_resps = items[1]
    # filtered_resps is a list like ["D"]
    if isinstance(filtered_resps, list):
        predicted = filtered_resps[0] if filtered_resps else ""
    else:
        predicted = str(filtered_resps)
    # Normalize
    gold = str(gold).upper().strip()
    predicted = str(predicted).upper().strip()
    # Take first character if longer
    if predicted:
        predicted = predicted[0]
    return 1.0 if predicted == gold else 0.0


def strict_acc(items):
    """
    Train/test split accuracy for generative multiple choice problems.
    - items[0] (target): str like "A" or "A,B" (multiple valid answers)
    - items[1] (filtered_resps): list like ["A", "B", "C", "D"]
    """
    target = items[0]
    correct_answers = set(re.split(r"[,;| ]+", target))

    filtered_resps = items[1][0]
    if not filtered_resps and not isinstance(filtered_resps, list):
        return 0.0
    predictions = set(re.split(r"[,;| ]+", filtered_resps))
    if not predictions:
        return 0.0

    # give 1 point if all correct answer is included in prediction
    if correct_answers == predictions:
        score = 1
    else:
        score = 0

    return score


def loose_acc(items):
    """
    SpatialEval accuracy for generative multiple choice problems.
    - items[0] (target): str like "A" or "A,B" (multiple valid answers)
    - items[1] (filtered_resps): list like ["A", "B", "C", "D"]
    """
    target = items[0]
    correct_answers = set(re.split(r"[,;| ]+", target))

    filtered_resps = items[1][0]
    if not filtered_resps and not isinstance(filtered_resps, list):
        return 0.0
    predictions = set(re.split(r"[,;| ]+", filtered_resps))
    if not predictions:
        return 0.0

    # give 1 point if all correct answer is included in prediction
    if correct_answers.issubset(predictions):
        score = 1
    else:
        score = 0

    return score
