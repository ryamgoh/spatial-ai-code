"""Historical transfer results retain paired effects and archive integrity."""

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
from spatial.v2.study.summarize_randomized import digest
from spatial.v2.study.summarize_transfer_diagnostic import paired_effects, summarize


class TransferDiagnosticTests(unittest.TestCase):
    def test_paired_effect_direction(self):
        rows = [
            {
                "base_id": f"{kind}-{status}",
                "query_kind": kind,
                "status": status,
                "answer_mode": mode,
                "layout": layout,
                "strict_acc": int(mode == "all-possible"),
            }
            for kind in ("direction", "which", "count")
            for status in ("unique", "ambiguous")
            for mode in ("single", "all-possible")
            for layout in ("paragraph", "lines")
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
            historical_root = Path(folder) / "source"
            required = set(receipt["environment_sha256"]) | {
                job["config"] for job in receipt["jobs"]
            }
            # Reconstruct only named inputs; never bypass frozen hash checks or
            # extract an archive over the working tree.
            with tarfile.open(snapshot) as archive:
                for name in required:
                    relative = Path(name)
                    self.assertFalse(relative.is_absolute() or ".." in relative.parts)
                    target = historical_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if (root / relative).is_file():
                        content = (root / relative).read_bytes()
                    else:
                        source = archive.extractfile(name)
                        self.assertIsNotNone(source)
                        with source:
                            content = source.read()
                    target.write_bytes(content)

            def write_archive():
                with tarfile.open(path, "w:gz") as archive:
                    for name, content in members.items():
                        info = tarfile.TarInfo(name)
                        info.size = len(content)
                        archive.addfile(info, io.BytesIO(content))

            write_archive()
            report = summarize(path, prepared, historical_root)
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
                summarize(path, prepared, historical_root)


if __name__ == "__main__":
    unittest.main()
