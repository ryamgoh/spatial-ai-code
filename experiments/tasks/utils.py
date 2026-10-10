import random
import re


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


# Census of data/spatialeval_cleaned.jsonl (1500 SpatialMap TQA rows).
# Empty gold = clean_v5.solve returned "No valid options found" and
# oracle_option was wiped (not a difficulty drop). Counted on the jsonl:
#   empty oracle                          171
#     dir     75   both axes unknown, or remaining dirs not in A–D
#     count   96   definite count is not one of the four numeric options
#     which    0   type-1 fallback keeps every option whose name is in
#                  the passage (that is which-4-ans, not empty)
#   SpatialMap-TQA-Corr (nonempty)       1329
#     Single (exactly one A–D letter)    1038
#       count 1-ans  404
#       dir   1-ans  332
#       which 1-ans  302
#     Multi (2+ letters)                  291
#       dir   2-ans   93
#       which 4-ans  198
# dir-4-ans / which-2-ans / count-multi do not occur in TQA-Corr.
#
# SpatialMap-TQA-Corr-Full (data/spatialeval_corr_full.jsonl, 1500):
#   same 1500 rows + option "E. None of these is proven" on every item.
#   gold E (369): empty oracle dir 75 + count 96, plus which-4 fallback 198
#     (type-1 first pass proves 0 of A–D; A,B,C,D was not a spatial gold)
#   remaining 1131 → original A–D gold; E is a distractor


def filter_corr_full(dataset):
    """SpatialMap-TQA-Corr-Full: 1500 rows with option E; gold is A–E letters."""
    def keep(doc):
        ls = _oracle_letters(doc)
        return bool(ls) and all(x in "ABCDE" for x in ls)

    return dataset.filter(keep)


def process_docs_v6_sft(dataset):
    """SFT chat jsonl → lm-eval docs. Gold letters A–E from Answer: line."""
    def convert(doc):
        user_content = ""
        oracle = ""
        for msg in doc.get("messages") or []:
            if msg.get("role") == "user":
                user_content = msg.get("content") or ""
            elif msg.get("role") == "assistant":
                match = re.search(
                    r"Answer:\s*([A-E](?:\s*,\s*[A-E])*)",
                    msg.get("content") or "",
                )
                if match:
                    oracle = ",".join(
                        p.strip() for p in match.group(1).split(",") if p.strip()
                    )
        return {"text": user_content, "oracle_option": oracle}

    return dataset.map(convert)


def process_docs_v2_sft(dataset):
    """Spatial V2 chat rows → lm-eval docs using solver-authored metadata gold."""
    from spatial.v2.context_budget import validate_context_admission

    def convert(doc):
        validate_context_admission(doc)
        user_content = next(
            (
                str(message.get("content") or "")
                for message in doc.get("messages") or []
                if message.get("role") == "user"
            ),
            "",
        )
        metadata = dict(doc.get("metadata") or {})
        evaluation_prompt = metadata["evaluation_prompt"]
        letters = [
            str(letter).strip().upper()
            for letter in metadata.get("oracle_letters") or []
            if str(letter).strip()
        ]
        if (
            not letters
            or letters != sorted(set(letters))
            or any(
                len(letter) != 1 or letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                for letter in letters
            )
        ):
            raise ValueError("Spatial V2 row has invalid metadata.oracle_letters")
        return {
            "text": user_content,
            "oracle_option": ",".join(letters),
            "evaluation_prompt": evaluation_prompt,
            "matrix_cell": str(metadata.get("matrix_cell") or ""),
            "answer_mode": str(metadata.get("answer_mode") or ""),
            "trace_format": str(metadata.get("trace_format") or ""),
            "supervision_arm": str(metadata.get("supervision_arm") or "checked-trace"),
            "difficulty": dict(metadata.get("difficulty") or {}),
        }

    return dataset.map(convert)


def applicable_process_rate(items):
    """Conditional replay rate; null means this format has no process checker."""
    applicable = sum(item[1] for item in items)
    return sum(item[0] for item in items) / applicable if applicable else None


def process_results_v2(doc, results):
    """Score the final answer and replay emitted Symbolic evidence separately."""
    response = results[0]
    footer = re.search(r"(?:^|\n)Answer:\s*([A-Z](?:\s*,\s*[A-Z])*)\s*$", response)
    letters = (
        tuple(part.strip() for part in footer.group(1).split(",")) if footer else ()
    )
    canonical = bool(letters) and letters == tuple(sorted(set(letters)))
    predicted = set(letters) if canonical else set()
    gold = _answer_letter_set(doc["oracle_option"])
    exact = int(bool(predicted) and predicted == gold)
    applicable = (
        doc["trace_format"] == "symbolic" and doc["supervision_arm"] != "answer-only"
    )
    metrics = {
        "strict_acc": exact,
        "loose_acc": int(bool(predicted) and gold.issubset(predicted)),
        "process_applicable": int(applicable),
        **{
            name: (0, int(applicable))
            for name in (
                "reasoning_valid",
                "decision_valid",
                "domain_consistent",
                "fully_valid",
            )
        },
    }
    if not applicable:
        return metrics

    from spatial.v2.grading import AnswerMode, encode_menu_answer, resolve_answer
    from spatial.v2.solver import SpatialSolverV2
    from spatial.v2.symbolic_trace_codec import score_symbolic_training_trace
    from spatial.v2.text import SpatialTextAdapter

    parsed = SpatialTextAdapter().parse(doc["text"])
    analysis = SpatialSolverV2(backend="z3").analyze(parsed.problem)
    expected = encode_menu_answer(
        resolve_answer(analysis, AnswerMode(doc["answer_mode"])), parsed.options
    )
    if expected.letters != frozenset(gold):
        raise ValueError(
            "evaluation metadata gold disagrees with visible-premise semantics"
        )
    # Some chat templates prefill <think>; its opening tag is then outside the response.
    if response.count("</think>") != 1:
        return metrics
    reasoning, answer_tail = response.split("</think>", 1)
    reasoning = reasoning.strip()
    if reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>") :].strip()
    score = score_symbolic_training_trace(parsed.problem, reasoning, expected)
    metrics.update(
        {
            name: (int(getattr(score, name)), 1)
            for name in ("reasoning_valid", "decision_valid", "domain_consistent")
        }
    )
    clean_footer = re.fullmatch(r"\s*Answer:\s*[A-Z](?:\s*,\s*[A-Z])*\s*", answer_tail)
    metrics["fully_valid"] = (
        int(score.fully_valid and exact and bool(clean_footer)),
        1,
    )
    return metrics


def process_docs_v13_sft(dataset):
    """Synthetic v13 chat JSONL → lm-eval docs with difficulty metadata.

    Unlike the legacy v6 adapter, v13 rows already carry solver-verified gold
    at the top level.  Preserve that gold and the structural analysis instead
    of re-extracting an answer from the supervised assistant trace.
    """

    def convert(doc):
        user_content = ""
        for msg in doc.get("messages") or []:
            if msg.get("role") == "user":
                user_content = str(msg.get("content") or "")
                break
        difficulty = dict(doc.get("difficulty") or {})
        semantic_subtype = str(
            doc.get("semantic_subtype")
            or doc.get("base_semantic_subtype")
            or difficulty.get("semantic_subtype")
            or ""
        )
        # Schema <=3 used derived *-cycle labels in difficulty while retaining
        # the real base subtype separately. Normalize old generated rows at the
        # evaluation boundary so historical results remain analysable.
        if semantic_subtype and (
            not difficulty.get("semantic_subtype")
            or str(difficulty.get("semantic_subtype")).endswith("-cycle")
        ):
            difficulty["semantic_subtype"] = semantic_subtype
        return {
            "text": user_content,
            "oracle_option": str(doc.get("oracle_option") or ""),
            "semantic_subtype": semantic_subtype,
            "difficulty": difficulty,
            "difficulty_schema_version": int(
                doc.get("difficulty_schema_version") or 1
            ),
            "generator_version": str(doc.get("generator_version") or ""),
            "generation_cell": str(doc.get("generation_cell") or ""),
        }

    return dataset.map(convert)


def filter_v6_spatialmap(dataset):
    """SpatialMap with v6 fifth option; keep nonempty A–E gold."""
    def keep(doc):
        ls = _oracle_letters(doc)
        return bool(ls) and all(x in "ABCDE" for x in ls)

    return dataset.filter(keep)


def filter_nonempty_oracle(dataset):
    """Drop rows whose gold letter set is empty (171 of 1500 cleaned SpatialMap)."""
    return dataset.filter(
        lambda doc: bool(str(doc.get("oracle_option") or "").strip())
    )


def _oracle_letters(doc) -> list[str]:
    raw = str(doc.get("oracle_option") or "").strip().upper()
    if not raw:
        return []
    return [p for p in re.split(r"[,;| ]+", raw) if p]


def filter_single_letter_oracle(dataset):
    """SpatialMap-TQA-Corr-Single: gold is exactly one A–D letter (1038 of 1329)."""
    def keep(doc):
        ls = _oracle_letters(doc)
        return len(ls) == 1 and ls[0] in "ABCD"

    return dataset.filter(keep)


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
    correct_answers = _answer_letter_set(target)

    filtered_resps = items[1][0]
    if not filtered_resps and not isinstance(filtered_resps, list):
        return 0.0
    predictions = _answer_letter_set(filtered_resps)
    if not predictions:
        return 0.0

    return 1 if correct_answers == predictions else 0


def loose_acc(items):
    """
    SpatialEval accuracy for generative multiple choice problems.
    - items[0] (target): str like "A" or "A,B" (multiple valid answers)
    - items[1] (filtered_resps): list like ["A", "B", "C", "D"]
    """
    target = items[0]
    correct_answers = _answer_letter_set(target)

    filtered_resps = items[1][0]
    if not filtered_resps and not isinstance(filtered_resps, list):
        return 0.0
    predictions = _answer_letter_set(filtered_resps)
    if not predictions:
        return 0.0

    return 1 if correct_answers.issubset(predictions) else 0


def _answer_letter_set(value):
    return {
        token
        for token in re.split(r"[,;| ]+", str(value).upper())
        if len(token) == 1 and "A" <= token <= "Z"
    }
