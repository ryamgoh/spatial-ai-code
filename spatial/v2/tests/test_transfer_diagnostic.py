"""Paired diagnostic labels and layouts must preserve the solved problem."""

import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.tasks.utils import process_results_v2_answer
from spatial.v2.study.analyze_randomized import archive_digest
from spatial.v2.study.prepare_transfer_diagnostic import generate, preparation_source_hashes
from spatial.v2.study.summarize_randomized import digest
from spatial.v2.study.summarize_transfer_diagnostic import paired_effects, summarize
from spatial.v2.text import SpatialTextAdapter


class TransferDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.audits, cls.rejected = generate(1, 890026, set())

    def test_balanced_complete_pairing(self):
        self.assertEqual(len(self.rows), 24)
        self.assertEqual(len(self.audits), 6)
        self.assertEqual(len({a["structure_signature"] for a in self.audits}), 6)
        self.assertEqual({a["solver"] for a in self.audits}, {"z3"})
        self.assertEqual(self.rejected, 0)

    def test_only_policy_and_layout_change(self):
        parser = SpatialTextAdapter()
        for audit in self.audits:
            views = [r for r in self.rows if r["base_id"] == audit["base_id"]]
            parsed = [parser.parse(r["text"]) for r in views]
            self.assertTrue(all(p.problem == parsed[0].problem for p in parsed))
            self.assertTrue(all(p.options == audit["options"] for p in parsed))
            self.assertTrue(
                all(p.options["E"] == "Cannot be determined" for p in parsed)
            )
            for mode in ("single", "all-possible"):
                a, b = [r for r in views if r["answer_mode"] == mode]
                self.assertEqual(a["oracle_option"], b["oracle_option"])
                self.assertEqual(a["text"].split(), b["text"].split())
                gold = set(a["oracle_option"].split(","))
                if mode == "single" and audit["status"] == "ambiguous":
                    self.assertEqual(gold, {"E"})
                else:
                    self.assertNotIn("E", gold)
                    self.assertEqual(len(gold), len(audit["possible_values"]))
            self.assertTrue(
                all("possible_values" not in r["evaluation_prompt"] for r in views)
            )

    def test_exclusion_and_determinism(self):
        forbidden = {self.audits[0]["structure_signature"]}
        rows, audits, rejected = generate(1, 890026, forbidden)
        self.assertGreaterEqual(rejected, 1)
        self.assertFalse(forbidden & {a["structure_signature"] for a in audits})
        repeated, _, _ = generate(1, 890026, forbidden)
        self.assertEqual(rows, repeated)

    def test_paired_effect_direction(self):
        rows = [
            {**r, "strict_acc": int(r["answer_mode"] == "all-possible")}
            for r in self.rows
        ]
        effects = paired_effects(rows)
        self.assertTrue(
            all(e["difference_pp"] == -100 for e in effects if e["factor"] == "policy")
        )
        self.assertTrue(
            all(e["difference_pp"] == 0 for e in effects if e["factor"] == "layout")
        )
        with self.assertRaises(ValueError):
            paired_effects(rows[:-1])
        with self.assertRaises(ValueError):
            paired_effects(rows + [rows[0]])

    def test_prepared_package_and_archive_replay(self):
        root = Path(__file__).resolve().parents[3]
        prepared = (
            root
            / "experiments/spatialentail-randomized-v2/prepared-transfer-diagnostic"
        )
        receipt = json.loads((prepared / "jobs.json").read_text())
        self.assertEqual(receipt["unique_bases"], 60)
        self.assertEqual(receipt["rows_per_model"], 240)
        for path, expected in receipt["file_sha256"].items():
            self.assertEqual(digest(prepared / path), expected)
        # Frozen preparation provenance belongs to its source snapshot, not the
        # current working tree after the curriculum namespace migration.
        snapshot = root / "experiments/archive/precomposition-source-2372c13.tar.gz"
        with tarfile.open(snapshot) as archive:
            for path, expected in receipt["preparation_source_sha256"].items():
                self.assertEqual(archive_digest(archive, path), expected)
        members = {"jobs.json": (prepared / "jobs.json").read_bytes()}
        for job in receipt["jobs"]:
            subprocess.run(["bash", "-n", str(root / job["script"])], check=True)
            config = yaml.safe_load((root / job["config"]).read_text())
            self.assertTrue(config["model_args"]["allow_multiple_answers"])
            prefix = f"results/{job['name']}/"
            payload = {"config": {**config, "limit": None}, "results": {}}
            for task in receipt["tasks"]:
                samples = []
                for row in map(
                    json.loads, (prepared / task["data"]).read_text().splitlines()
                ):
                    response = f"<stage1_response>\nAnswer: {row['oracle_option']}\n</stage1_response>\n\nAnswer: {row['oracle_option']}"
                    samples.append(
                        {
                            "doc": row,
                            "target": row["oracle_option"],
                            "arguments": [
                                [
                                    row["evaluation_prompt"],
                                    {
                                        "max_gen_toks": 4096,
                                        "temperature": 0,
                                        "repetition_penalty": 1,
                                        "until": [],
                                    },
                                ]
                            ],
                            "resps": [[response]],
                            "filtered_resps": [response],
                            **process_results_v2_answer(row, [response]),
                        }
                    )
                self.assertEqual(len(samples), 60)
                payload["results"][task["task"]] = {
                    "sample_len": len(samples),
                    "strict_acc": 1,
                    "stage1_acc": 1,
                    "stage1_format_valid": 1,
                }
                members[prefix + f"responses_{task['task']}.jsonl"] = "".join(
                    json.dumps(s) + "\n" for s in samples
                ).encode()
            members[prefix + "results.json"] = json.dumps(payload).encode()
            members[prefix + "COMPLETED"] = b""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.tar.gz"

            def write_archive():
                with tarfile.open(path, "w:gz") as archive:
                    for name, content in members.items():
                        info = tarfile.TarInfo(name)
                        info.size = len(content)
                        archive.addfile(info, io.BytesIO(content))

            write_archive()
            report = summarize(path, prepared, root)
            self.assertEqual(len(report["models"]), 7)
            self.assertTrue(
                all(
                    r["difference_pp"] == 0
                    for m in report["models"].values()
                    for r in m["paired_effects"]
                )
            )
            response_key = next(k for k in members if "/responses_" in k)
            members[response_key] += members[response_key].splitlines(keepends=True)[0]
            write_archive()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                summarize(path, prepared, root)

    def test_new_preparation_hashes_include_legacy_dependencies(self):
        root = Path(__file__).resolve().parents[3]
        hashes = preparation_source_hashes(root)
        for name in ("generation", "possibility", "benchmark_registry"):
            path = f"spatial/v2/legacy/{name}.py"
            self.assertEqual(hashes[path], digest(root / path))


if __name__ == "__main__":
    unittest.main()
