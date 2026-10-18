from copy import deepcopy
import json

import yaml

from spatial.v2.study.comparator_evaluation import (
    prepare_general_instruction_evaluation,
)
from spatial.v2.study.randomized_v2 import _headline_row
from spatial.v2.study.reasoning_evaluation import (
    REASONING_MODELS,
    prepare_reasoning_evaluation,
)


def test_reasoning_batch_has_the_four_agreed_conditions() -> None:
    assert [spec.model.name for spec in REASONING_MODELS] == [
        "qwen3-5-4b-thinking-on",
        "qwen3-5-9b-thinking-on",
        "ministral-3-8b-reasoning",
        "deepseek-r1-distill-qwen-7b",
    ]
    assert all(len(spec.model.revision) == 40 for spec in REASONING_MODELS)


def test_reasoning_package_reuses_general_data_with_model_sampling(tmp_path) -> None:
    benchmark = tmp_path / "benchmark.jsonl"
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
    benchmark.write_text("".join(json.dumps(row) + "\n" for row in rows))
    general = tmp_path / "prepared-general-eval"
    prepare_general_instruction_evaluation(benchmark, general, repo_root=tmp_path)
    output = tmp_path / "prepared-reasoning-eval"

    receipt = prepare_reasoning_evaluation(
        general,
        output,
        repo_root=tmp_path,
        expected_total_bases=48,
    )

    assert receipt["total_bases"] == 48
    assert receipt["tasks_per_model"] == 48
    assert len(receipt["jobs"]) == 4
    assert not (output / "data").exists()
    qwen_task = next(
        (output / "tasks").glob("*qwen3_5_9b_thinking_on*.yaml")
    ).read_text()
    for setting in (
        "max_gen_toks: 4096",
        "temperature: 1.0",
        "top_p: 0.95",
        "repetition_penalty: 1",
        "seed: 42",
        "top_k: 20",
    ):
        assert setting in qwen_task
    smoke = yaml.safe_load(
        (output / "configs/smoke/deepseek-r1-distill-qwen-7b.yaml").read_text()
    )
    assert smoke["limit"] == 1
    assert (output / "submit-h200-smoke-chain.sh").is_file()
