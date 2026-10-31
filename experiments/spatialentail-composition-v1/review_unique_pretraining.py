"""Read-only prose census and training-only, family-held-out shortcut diagnostic."""

import argparse
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def features(record):
    """Only visible scene size/edge counts; no solver or answer metadata."""
    problem = record["problem"]
    objects = problem["objects"]
    atoms = problem["premise"]["operands"]
    degree = Counter()
    directions = Counter()
    pairs = set()
    for atom in atoms:
        a, b = atom["subject"], atom["reference"]
        degree[a] += 1
        degree[b] += 1
        pairs.add(tuple(sorted((a, b))))
        directions.update(atom["allowed"])
    return [
        len(objects),
        len(atoms),
        len(pairs),
        max(degree.values(), default=0),
        sum(degree[o] == 1 for o in objects),
        *[
            directions[d]
            for d in (
                "North",
                "Northeast",
                "East",
                "Southeast",
                "South",
                "Southwest",
                "West",
                "Northwest",
            )
        ],
    ]


def main():
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
    )
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.tree import DecisionTreeClassifier, export_text

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh report file")
    manifest_path = args.candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["payload_sha256"].items():
        if digest(args.candidate / name) != expected:
            raise ValueError("candidate payload hash mismatch")
    with (args.candidate / "evidence.jsonl").open() as stream:
        records = [json.loads(line) for line in stream]
    if len(records) != 6000 or any(r["split"] != "train" for r in records):
        raise ValueError("expected 6,000 training-only records")
    phrases = (
        "Unique requires exactly one",
        "every contract requires E",
        "No minimality claim is made",
        "The complete entailed-answer set",
    )
    prose, source_counts, redundant = Counter(), Counter(), Counter()
    samples = {}
    for record in records:
        row = record["rows"][0]
        kind = record["query_kind"]
        cat = "A-D" if row["answer"] in "ABCD" else row["escape_reason"]
        source_counts[kind, record["source"], cat] += 1
        for style, text in row["references"].items():
            for phrase in phrases:
                if phrase in text:
                    prose[style, phrase] += 1
            if kind in ("direction", "count") and cat == "A-D":
                redundant[kind, style, "rows"] += 1
                redundant[kind, style, "negative_claim_lines"] += text.count(
                    "so it is not entailed."
                )
        key = f"{kind}/{cat}"
        if key not in samples:
            samples[key] = {
                "id": row["id"],
                "prompt": row["prompt"],
                "references": row["references"],
            }
    # Union family and story identities before splitting, across query types.
    parents = {}

    def root(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = root(parents[x])
        return parents[x]

    for record in records:
        a, b = (
            root("family:" + record["family_id"]),
            root("story:" + record["story_id"]),
        )
        parents[b] = a
    all_groups = [root("family:" + r["family_id"]) for r in records]
    results = []
    feature_names = [
        "objects",
        "premises",
        "distinct_pairs",
        "max_degree",
        "leaf_objects",
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
    ]
    for kind in ("direction", "which", "count"):
        indices = [i for i, r in enumerate(records) if r["query_kind"] == kind]
        y = np.array(
            [records[i]["rows"][0]["answer"] == "E" for i in indices], dtype=int
        )
        groups = np.array([all_groups[i] for i in indices])
        all_x = np.array([features(records[i]) for i in indices])
        for name, columns in (
            ("size_only_depth_1", 2),
            ("structural_counts_depth_3", len(feature_names)),
        ):
            x = all_x[:, :columns]
            predicted = np.empty(len(y), dtype=int)
            folds = []
            splitter = StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=31031
            )
            for train, test in splitter.split(x, y, groups):
                if set(groups[train]) & set(groups[test]):
                    raise ValueError("family/story leakage across diagnostic folds")
                model = DecisionTreeClassifier(
                    max_depth=1 if columns == 2 else 3,
                    min_samples_leaf=25,
                    class_weight="balanced",
                    random_state=31031,
                )
                model.fit(x[train], y[train])
                predicted[test] = model.predict(x[test])
                folds.append(
                    {
                        "train": len(train),
                        "test": len(test),
                        "rule": export_text(
                            model, feature_names=feature_names[:columns]
                        ),
                    }
                )
            results.append(
                {
                    "query": kind,
                    "model": name,
                    "examples": len(y),
                    "accuracy": accuracy_score(y, predicted),
                    "balanced_accuracy": balanced_accuracy_score(y, predicted),
                    "confusion_matrix_nonE_E": confusion_matrix(
                        y, predicted, labels=[0, 1]
                    ).tolist(),
                    "always_nonE_accuracy": float((y == 0).mean()),
                    "always_nonE_balanced_accuracy": 0.5,
                    "folds": folds,
                }
            )
    report = {
        "status": "diagnostic-complete-not-training-release",
        "candidate_manifest_sha256": digest(manifest_path),
        "diagnostic_sha256": digest(Path(__file__)),
        "sklearn_version": version("scikit-learn"),
        "method": (
            "Five family-and-story-group-held-out folds within training, fixed seeds, "
            "two prespecified shallow tree baselines predicting E versus non-E. "
            "No evaluation data, IDs, source labels, depth metadata, menus, proof text, "
            "semantic metadata or solver outputs used as features. Labels used as targets only."
        ),
        "limitations": (
            "This measures a shortcut opportunity, not SFT model behavior or full A-F accuracy. "
            "Family/story grouping is not proof of complete structural independence. "
            "No evaluation set or GPU model was used."
        ),
        "prose_phrase_counts": [
            {"style": s, "phrase": p, "rows": n} for (s, p), n in sorted(prose.items())
        ],
        "redundant_exact_value_sections": [
            {"query": q, "style": s, "metric": m, "count": n}
            for (q, s, m), n in sorted(redundant.items())
        ],
        "source_answer_counts": [
            {"query": q, "source": s, "category": c, "count": n}
            for (q, s, c), n in sorted(source_counts.items())
        ],
        "shortcut_results": results,
        "sample_references": samples,
        "dataset_modified": False,
        "training_started": False,
    }
    for name, expected in manifest["payload_sha256"].items():
        if digest(args.candidate / name) != expected:
            raise ValueError("payload changed during review")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("sample_references", "shortcut_results")
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            [{k: v for k, v in result.items() if k != "folds"} for result in results],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
