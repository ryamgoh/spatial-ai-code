"""Corr summaries reject incomplete or altered evaluation evidence."""

import copy
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.tasks.utils import process_results_v2_answer
from spatial.v2.study.summarize_corr import SCHEMA, describe, inspect_sample, summarize
from spatial.v2.study.summarize_randomized import digest


class CorrSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared-corr-eval"
        self.prepared.mkdir()
        self.generation = {
            "max_gen_toks": 4096,
            "temperature": 0.0,
            "repetition_penalty": 1,
            "until": [],
        }
        self.source = {
            "id": "example",
            "oracle_option": "E",
            "query_kind": "direction",
            "audit_status": "underdetermined-oracle-possible",
            "answer_mode": "single",
            "trace_format": "natural",
            "supervision_arm": "answer-only",
            "evaluation_prompt": "Which option?",
        }
        self.sample = self.make_sample(self.source, "A", "E")

    def make_sample(self, source, native, final):
        response = f"<stage1_response>\nAnswer: {native}\n</stage1_response>\n\nAnswer: {final}"
        return {
            "doc": source,
            "target": source["oracle_option"],
            "arguments": [[source["evaluation_prompt"], self.generation]],
            "resps": [[response]],
            "filtered_resps": [response],
            **process_results_v2_answer(source, [response]),
        }

    def inspect(self, sample):
        return inspect_sample(
            sample, self.source, process_results_v2_answer, self.generation
        )

    def test_replay_and_abstention_denominators(self):
        row = self.inspect(self.sample)
        self.assertEqual(row["strict_acc"], 1)
        self.assertEqual(row["stage1_acc"], 0)
        other = {
            **row,
            "id": "determinate",
            "status": "determinate",
            "gold": "A",
            "strict_acc": 0,
        }
        result = describe([row, other])
        self.assertEqual(result["e_recall"], 1)
        self.assertEqual(result["false_abstention"], 1)
        self.assertEqual(result["strict_acc"], 0.5)

    def test_sample_tampering_rejected(self):
        for field in (
            "doc",
            "target",
            "arguments",
            "strict_acc",
            "resps",
            "filtered_resps",
        ):
            sample = copy.deepcopy(self.sample)
            if field == "doc":
                sample[field]["evaluation_prompt"] = "Altered"
            elif field == "arguments":
                sample[field][0][1]["max_gen_toks"] = 10
            elif field in ("resps", "filtered_resps"):
                sample[field] = []
            else:
                sample[field] = "A" if field == "target" else 0
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.inspect(sample)

    def test_multi_option_stage2_rejected(self):
        with self.assertRaisesRegex(ValueError, "SINGLE"):
            self.inspect(self.make_sample(self.source, "E", "A,E"))

    def fixture(self):
        root = Path(__file__).resolve().parents[3]
        upstream = self.root / "prepared-eval"
        upstream.mkdir()
        (upstream / "jobs.json").write_text("{}")
        rows = [
            self.source,
            {
                **self.source,
                "id": "second",
                "oracle_option": "A",
                "audit_status": "exact-match",
            },
        ]
        samples = [
            self.make_sample(r, r["oracle_option"], r["oracle_option"]) for r in rows
        ]
        for folder in ("data", "tasks", "configs"):
            (self.prepared / folder).mkdir()
        (self.prepared / "data/corr.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        (self.prepared / "tasks/corr.yaml").write_text(
            yaml.safe_dump({"generation_kwargs": self.generation})
        )
        (self.prepared / "correction-summary.json").write_text(
            json.dumps({"outputs": {"corrected": {"sha256": "correction"}}})
        )
        names = ["untuned"] + [
            f"{arm}-seed-{seed}"
            for arm in ("single-natural", "single-answer-only", "multi-natural")
            for seed in (42, 43)
        ]
        config = {"model": "vllm_staged_pass", "model_args": {"pretrained": "fixture"}}
        jobs = []
        members = {}
        for name in names:
            path = self.prepared / f"configs/{name}.yaml"
            path.write_text(yaml.safe_dump(config))
            jobs.append({"name": name, "config": str(path)})
            payload = {
                "config": {**config, "limit": None},
                "results": {
                    "corr": {
                        "sample_len": 2,
                        "strict_acc": 1,
                        "stage1_acc": 1,
                        "stage1_format_valid": 1,
                    }
                },
            }
            prefix = f"results/{name}/"
            members[prefix + "COMPLETED"] = b""
            members[prefix + "results.json"] = json.dumps(payload).encode()
            members[prefix + "responses_corr.jsonl"] = "".join(
                json.dumps(s) + "\n" for s in samples
            ).encode()
        receipt = {
            "schema": SCHEMA,
            "total_bases": 2,
            "gold_counts": {"A": 1, "E": 1},
            "corrected_sha256": "correction",
            "jobs": jobs,
            "tasks": [{"task": "corr", "count": 2}],
            "source_sft_evaluation_receipt_sha256": digest(upstream / "jobs.json"),
            "environment_sha256": {
                "experiments/tasks/utils.py": digest(
                    root / "experiments/tasks/utils.py"
                )
            },
            "file_sha256": {
                str(p.relative_to(self.prepared)): digest(p)
                for p in self.prepared.rglob("*")
                if p.is_file()
            },
        }
        content = json.dumps(receipt).encode()
        (self.prepared / "jobs.json").write_bytes(content)
        members["jobs.json"] = content
        return root, members

    def run_archive(self, root, members):
        archive = self.root / "results.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return summarize(archive, self.prepared, root, repetitions=100)

    def test_complete_archive_and_pairing(self):
        root, members = self.fixture()
        report, rows = self.run_archive(root, members)
        self.assertEqual(report["evaluated_responses"], 14)
        self.assertEqual(len(rows), 7)
        self.assertEqual(len(report["paired_comparisons"]), 30)
        self.assertTrue(
            all(r["difference_pp"] == 0 for r in report["paired_comparisons"])
        )

    def test_archive_failures(self):
        root, original = self.fixture()
        response = "results/untuned/responses_corr.jsonl"
        for case in (
            "receipt",
            "marker",
            "duplicate",
            "missing",
            "aggregate",
            "runtime",
            "unsafe",
        ):
            members = dict(original)
            if case == "receipt":
                members["jobs.json"] = b"{}"
            elif case == "marker":
                del members["results/untuned/COMPLETED"]
            elif case == "duplicate":
                members[response] += members[response].splitlines(keepends=True)[0]
            elif case == "missing":
                members[response] = members[response].splitlines(keepends=True)[0]
            elif case in ("aggregate", "runtime"):
                key = "results/untuned/results.json"
                payload = json.loads(members[key])
                if case == "aggregate":
                    payload["results"]["corr"]["strict_acc"] = 0
                else:
                    payload["config"]["model_args"]["pretrained"] = "wrong"
                members[key] = json.dumps(payload).encode()
            else:
                members["../outside"] = b""
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.run_archive(root, members)


if __name__ == "__main__":
    unittest.main()
