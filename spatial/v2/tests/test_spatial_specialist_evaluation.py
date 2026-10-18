from copy import deepcopy
import json

import yaml

from spatial.v2.study.comparator_evaluation import (
    prepare_general_instruction_evaluation,
)
from spatial.v2.study.randomized_v2 import _headline_row
from spatial.v2.study.spatial_specialist_evaluation import (
    SPATIAL_SPECIALISTS,
    prepare_spatial_specialist_evaluation,
)


def _general_package(tmp_path):
    template = _headline_row("direction", 1, "matched", 8, 0, 42)
    rows = []
    for query in ("direction", "which", "count"):
        for gold in range(1, 5):
            for partition in (
                "matched",
                "length-ood",
                "noise-ood",
                "composition-ood",
            ):
                row = deepcopy(template)
                row["id"] = f"test-{query}-g{gold}-{partition}"
                row["metadata"].update(
                    query_kind=query,
                    gold_size=gold,
                    benchmark_partition=partition,
                )
                rows.append(row)
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text("".join(json.dumps(row) + "\n" for row in rows))
    general = tmp_path / "prepared-general-eval"
    prepare_general_instruction_evaluation(benchmark, general, repo_root=tmp_path)
    return general


def test_spatial_specialist_panel_is_the_fixed_five() -> None:
    assert [spec.pretrained for spec in SPATIAL_SPECIALISTS] == [
        "YuhongLiu/SpatialASM-Qwen3.5-9B",
        "sensenova/SenseNova-SI-1.2-InternVL3-8B",
        "remyxai/SpaceQwen3-VL-2B-Thinking",
        "BAAI/RoboBrain2.5-8B-NV",
        "nvidia/Cosmos-Reason2-8B",
    ]
    assert all(len(spec.revision) == 40 for spec in SPATIAL_SPECIALISTS)


def test_spatial_package_reuses_text_tasks_and_has_three_gates(tmp_path) -> None:
    general = _general_package(tmp_path)
    output = tmp_path / "prepared-spatial-eval"

    receipt = prepare_spatial_specialist_evaluation(
        general,
        output,
        repo_root=tmp_path,
        expected_total_bases=48,
    )

    assert receipt["total_bases"] == 48
    assert receipt["tasks"] == 48
    assert len(receipt["jobs"]) == 5
    assert not (output / "data").exists()
    assert not (output / "tasks").exists()
    assert all(job["input_modality"] == "text-only" for job in receipt["jobs"])

    sense = yaml.safe_load(
        (output / "configs/sensenova-si-1-2-internvl3-8b.yaml").read_text()
    )
    assert sense["model"] == "vllm_staged_remote_chat_pass"
    preflight = yaml.safe_load(
        (
            output
            / "configs/preflight/sensenova-si-1-2-internvl3-8b.yaml"
        ).read_text()
    )
    assert len(preflight["tasks"]) == 1
    assert preflight["limit"] == 1
    smoke = yaml.safe_load(
        (output / "configs/smoke/cosmos-reason2-8b.yaml").read_text()
    )
    assert len(smoke["tasks"]) == 48
    assert smoke["limit"] == 1
    full = yaml.safe_load(
        (output / "configs/cosmos-reason2-8b.yaml").read_text()
    )
    assert len(full["tasks"]) == 48
    assert "limit" not in full
    assert (output / "submit-h200-preflight-chain.sh").is_file()
    assert (output / "submit-h200-smoke-chain.sh").is_file()
    assert (output / "submit-h200-chain.sh").is_file()
    preflight_submit = (output / "submit-h200-preflight-chain.sh").read_text()
    assert 'dependency=(--dependency="afterany:$job_id")' in preflight_submit
    smoke_submit = (output / "submit-h200-smoke-chain.sh").read_text()
    assert "results/preflight/cosmos-reason2-8b/COMPLETED" in smoke_submit
    full_submit = (output / "submit-h200-chain.sh").read_text()
    assert "results/smoke/cosmos-reason2-8b/COMPLETED" in full_submit
