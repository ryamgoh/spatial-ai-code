import random
import re
from clean_all import solve


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
