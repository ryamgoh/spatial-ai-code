"""CPU-only regression tests for downloaded evaluation aggregation."""

import json
import tempfile
import unittest
from pathlib import Path

from spatial.v2.study.summarize_randomized import (
    digest,
    metric,
    render,
    seed_summary,
    summarize,
    tasks_for,
)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "results" / "untuned"
        self.folder.mkdir(parents=True)
        (self.folder / "COMPLETED").touch()
        self.tasks = [
            {
                "task": "one",
                "count": 1,
                "query_kind": "count",
                "gold_size": 1,
                "partition": "matched",
            },
            {
                "task": "many",
                "count": 3,
                "query_kind": "count",
                "gold_size": 2,
                "partition": "noise-ood",
            },
        ]
        self.receipt = {
            "schema": "spatialentail-randomized-v2-eval-jobs-v1",
            "benchmark_sha256": "benchmark",
            "total_bases": 4,
            "tasks": self.tasks,
            "jobs": [{"name": "untuned"}],
        }
        self.payload = {"config": {"limit": None}, "results": {}}
        for task in self.tasks:
            correct = int(task["task"] == "one")
            values = {"strict_acc": correct, "stage1_acc": 0, "stage1_format_valid": 1}
            self.payload["results"][task["task"]] = {
                "sample_len": task["count"],
                **{key + ",none": value for key, value in values.items()},
            }
            samples = []
            for index in range(task["count"]):
                metadata = {
                    "base_id": f"{task['task']}-{index}",
                    "query_kind": task["query_kind"],
                    "gold_size": task["gold_size"],
                    "benchmark_partition": task["partition"],
                    "answer_mode": "single"
                    if task["gold_size"] == 1
                    else "all-possible",
                    "heldout_transitions": [],
                    "difficulty": {
                        "num_distractor_premises": index,
                        "proof_complexity": {"semantic_length": 2},
                    },
                }
                samples.append(
                    {
                        "doc": {
                            "metadata": metadata,
                            "oracle_option": "A" if correct else "A,B",
                        },
                        **values,
                    }
                )
            self.write(
                f"responses_{task['task']}.jsonl",
                "".join(json.dumps(sample) + "\n" for sample in samples),
            )
        (self.root / "jobs.json").write_text(json.dumps(self.receipt))
        self.save_results()

    def write(self, name, content):
        (self.folder / name).write_text(content)

    def save_results(self):
        self.write("results.json", json.dumps(self.payload))

    def test_weighting_contracts_and_dimensions(self):
        result = summarize(self.root)
        model = result["models"]["untuned"]
        self.assertEqual(model["overall"]["strict_acc"], 0.25)
        self.assertEqual(model["overall"]["partition_macro_acc"], 0.5)
        self.assertEqual(model["by_contract"]["single"]["strict_acc"], 1)
        self.assertEqual(model["by_contract"]["all-possible"]["n"], 3)
        self.assertEqual(model["overall"]["by_L"]["2"]["n"], 4)
        self.assertEqual(model["stage_outcomes"]["stage1_wrong_stage2_correct"], 1)
        self.assertIn("25.00", render({"test": result}))

    def test_missing_completion_is_incomplete(self):
        (self.folder / "COMPLETED").unlink()
        self.assertFalse(summarize(self.root)["models"]["untuned"]["complete"])

    def test_rejects_limited_run(self):
        self.payload["config"]["limit"] = 1
        self.save_results()
        with self.assertRaisesRegex(ValueError, "limited/smoke"):
            summarize(self.root)

    def test_rejects_wrong_result_count(self):
        self.payload["results"]["many"]["sample_len"] = 1
        self.save_results()
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            summarize(self.root)

    def test_rejects_missing_task(self):
        del self.payload["results"]["many"]
        self.save_results()
        with self.assertRaisesRegex(ValueError, "task coverage"):
            summarize(self.root)

    def test_rejects_sample_metric_disagreement(self):
        self.payload["results"]["many"]["strict_acc,none"] = 1
        self.save_results()
        with self.assertRaisesRegex(ValueError, "sample/result"):
            summarize(self.root)

    def test_rejects_duplicate_examples(self):
        path = self.folder / "responses_many.jsonl"
        first = path.read_text().splitlines()[0]
        path.write_text((first + "\n") * 3)
        with self.assertRaisesRegex(ValueError, "duplicate base IDs"):
            summarize(self.root)

    def test_rejects_truncated_responses(self):
        path = self.folder / "responses_many.jsonl"
        path.write_text(path.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "response count"):
            summarize(self.root)

    def test_rejects_nonfinite_metric(self):
        with self.assertRaises(ValueError):
            metric({"strict_acc": float("nan")}, "strict_acc")

    def test_reasoning_resolves_hash_bound_task_names(self):
        receipt = {
            "schema": "spatialentail-reasoning-eval-v1",
            "benchmark_sha256": "benchmark",
            "tasks_per_model": 2,
            "upstream_general_receipt_sha256": digest(self.root / "jobs.json"),
        }
        tasks = tasks_for(receipt, "qwen3-5-4b-thinking-on", self.root)
        self.assertEqual(
            tasks[1]["task"],
            "spatialentail_reasoning_qwen3_5_4b_thinking_on_count_g2_noise_ood",
        )
        receipt["upstream_general_receipt_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "receipt hash"):
            tasks_for(receipt, "model", self.root)

    def test_seed_means_and_ranges(self):
        model = summarize(self.root)["models"]["untuned"]
        models = {f"multi-natural-seed-{seed}": model for seed in (42, 43)}
        result = seed_summary(models)["multi-natural"]
        self.assertEqual(result["overall"]["mean"], 0.25)
        self.assertEqual(result["overall"]["range"], 0)
        del models["multi-natural-seed-43"]
        self.assertEqual(seed_summary(models), {})


if __name__ == "__main__":
    unittest.main()
