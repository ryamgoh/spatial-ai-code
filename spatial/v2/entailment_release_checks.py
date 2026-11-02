"""Reproducible, read-only evidence checks for an already generated candidate."""

import hashlib
import json
import multiprocessing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

from spatial.v2.entailment_artifact_audit import (
    audit_record,
    checked_sat,
    independent_claims,
    label,
    replay,
    require,
)
from spatial.v2.entailment_external import verify_view
from spatial.v2.entailment_generation import digest, read_jsonl
from spatial.v2.entailment_release import COUNTS
from spatial.v2.entailment_scene_protocol import parse_row


def nuisance_features(row, prompt_tokens):
    problem, menu, _ = parse_row(row)
    mentions = Counter(
        name
        for atom in problem.premise.operands
        for name in (atom.subject, atom.reference)
    )
    size = {
        "objects": len(problem.objects),
        "premises": len(problem.premise.operands),
        "prompt_chars": len(row["prompt"]),
        "prompt_tokens": prompt_tokens,
    }
    names = {f"present/{name}": 1 for name in problem.objects}
    names.update({f"mentions/{name}": value for name, value in mentions.items()})
    names[f"reference/{problem.query.reference}"] = 1
    if row["query_kind"] == "direction":
        names[f"target/{problem.query.target}"] = 1
    menus = {f"menu/{letter}/{value}": 1 for letter, value in zip("ABCD", menu)}
    return {
        "lengths": size,
        "names": names,
        "menus": menus,
        "combined": {**size, **names, **menus},
    }


def diagnose(artifacts):
    import numpy as np
    import sklearn
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GroupKFold
    from sklearn.tree import DecisionTreeClassifier

    require(sklearn.__version__ == "1.7.2", "diagnostics require scikit-learn 1.7.2")
    rows = read_jsonl(artifacts / "data/train/references.jsonl")
    require(len(rows) == COUNTS["train"], "diagnostics require full Train")
    tokens = {
        r["id"]: r["prompt_tokens"]
        for r in read_jsonl(artifacts / "data/train/token-measurements.jsonl")
        if r["style"] == "natural"
    }
    features = [nuisance_features(row, tokens[row["id"]]) for row in rows]
    report = {
        "schema": "spatialentail-nuisance-probe-v1",
        "manifest_sha256": digest(artifacts / "manifest.json"),
        "train_sha256": digest(artifacts / "data/train/references.jsonl"),
        "policy": {
            "data": "train-only",
            "folds": 5,
            "groups": "family_id",
            "seed": 42,
            "classifier": "DecisionTreeClassifier(max_depth=3,min_samples_leaf=30)",
            "investigate_lift": 0.10,
            "features": "input-only; no references, answers, case, IDs, or construction metadata",
            "sklearn": sklearn.__version__,
        },
        "results": [],
    }
    for kind in ("direction", "which", "count"):
        indices = [i for i, row in enumerate(rows) if row["query_kind"] == kind]
        selected = [rows[i] for i in indices]
        groups = np.array([row["family_id"] for row in selected])
        folds = list(
            GroupKFold(n_splits=5, shuffle=True, random_state=42).split(
                indices, groups=groups
            )
        )
        for target in ("letter", "answer_class"):
            y = np.array(
                [
                    row["answer"]
                    if target == "letter" or row["answer"] in "EF"
                    else "A-D"
                    for row in selected
                ]
            )
            baseline = np.empty(len(y), dtype=object)
            for train, test in folds:
                baseline[test] = Counter(y[train]).most_common(1)[0][0]
            for feature_set in ("lengths", "names", "menus", "combined"):
                predicted = np.empty(len(y), dtype=object)
                importances = Counter()
                for train, test in folds:
                    require(
                        not set(groups[train]) & set(groups[test]),
                        "diagnostic family leakage",
                    )
                    vectorizer = DictVectorizer(sparse=False)
                    xtrain = vectorizer.fit_transform(
                        [features[indices[i]][feature_set] for i in train]
                    )
                    xtest = vectorizer.transform(
                        [features[indices[i]][feature_set] for i in test]
                    )
                    model = DecisionTreeClassifier(
                        max_depth=3, min_samples_leaf=30, random_state=42
                    )
                    model.fit(xtrain, y[train])
                    predicted[test] = model.predict(xtest)
                    importances.update(
                        {
                            name: float(value) / 5
                            for name, value in zip(
                                vectorizer.feature_names_, model.feature_importances_
                            )
                        }
                    )
                accuracy = float(np.mean(predicted == y))
                majority = float(np.mean(baseline == y))
                labels = sorted(set(y))
                report["results"].append(
                    {
                        "query_kind": kind,
                        "target": target,
                        "feature_set": feature_set,
                        "n": len(y),
                        "families": len(set(groups)),
                        "accuracy": accuracy,
                        "majority_accuracy": majority,
                        "lift": accuracy - majority,
                        "investigate": accuracy - majority >= 0.10,
                        "labels": labels,
                        "confusion": confusion_matrix(
                            y, predicted, labels=labels
                        ).tolist(),
                        "top_features": importances.most_common(6),
                    }
                )
    report["status"] = (
        "investigation-required"
        if any(r["investigate"] for r in report["results"])
        else "no-trigger"
    )
    return report


def replay_one(item):
    import z3

    split, record = item
    row = record["rows"][0]
    if split == "external-test":
        verify_view(row)
        problem, menu, contract = parse_row(row)
        solver, claims = independent_claims(problem)
        consistent = checked_sat(solver)
        entailed = {
            value
            for value, claim in claims.items()
            if consistent and not checked_sat(solver, z3.Not(claim))
        }
        require(
            row["answer"] == label(consistent, entailed, menu, contract.value),
            "external label differs",
        )
        if record["evidence"] is not None:
            audit_record(
                {
                    **record,
                    **{
                        key: row[key]
                        for key in (
                            "interface",
                            "problem",
                            "prompt_version",
                            "scope_policy",
                            "study_scope",
                        )
                    },
                }
            )
    else:
        replay(record)
    return split, row["id"]


def replay_candidate(artifacts, workers):
    require(type(workers) is int and workers > 0, "workers must be positive")

    def records():
        for split in COUNTS:
            with (artifacts / f"data/{split}/evidence.jsonl").open() as source:
                for line in source:
                    yield split, json.loads(line)

    counts, ids = Counter(), []
    iterator = records()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        while batch := list(islice(iterator, workers * 4)):
            for split, identity in pool.map(replay_one, batch):
                counts[split] += 1
                ids.append(identity)
            print(f"Replayed {len(ids)}/{sum(COUNTS.values())}", flush=True)
    require(dict(counts) == COUNTS, "incomplete replay")
    return {
        "status": "passed",
        "manifest_sha256": digest(artifacts / "manifest.json"),
        "counts": dict(counts),
        "ordered_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "checks": [
            "fresh-independent-SMT-labels",
            "saved-counterexample-replay",
            "proof-and-citation-replay",
            "selected-depth-for-synthetic",
            "both-reference-renderings-where-present",
            "external-source-preservation",
        ],
        "review_type": "automated-replay-not-human-review",
    }


def data_profile(artifacts):
    import numpy as np

    result = {}
    for split in COUNTS:
        rows = read_jsonl(artifacts / f"data/{split}/references.jsonl")
        groups = defaultdict(list)
        by_id = {row["id"]: row for row in rows}
        for token in read_jsonl(artifacts / f"data/{split}/token-measurements.jsonl"):
            row = by_id[token["id"]]
            for field in (
                "prompt_tokens",
                "supervised_tokens",
                "training_tokens",
                "stage2_reserved_tokens",
            ):
                if field in token:
                    groups[f"{token['style']}/{field}"].append(token[field])
            key = f"{token['style']}/{row['query_kind']}/{row.get('construction_level')}/{row.get('case')}"
            groups[f"cell-prompt/{key}"].append(token["prompt_tokens"])
        result[split] = {
            "questions": len(rows),
            "answers": dict(Counter(f"{r['query_kind']}/{r['answer']}" for r in rows)),
            "tokens": {
                key: {
                    "n": len(values),
                    "min": min(values),
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "max": max(values),
                    "mean": float(np.mean(values)),
                }
                for key, values in groups.items()
            },
        }
    return {
        "status": "passed",
        "manifest_sha256": digest(artifacts / "manifest.json"),
        "splits": result,
    }
