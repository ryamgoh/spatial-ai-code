"""Four-run configuration, launch guards, masks and in-epoch selection checks."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
import generation_policy as policy

from spatial.v2 import entailment_sft as sft
from spatial.v2 import entailment_sft_runtime as runtime
from spatial.v2.entailment_generation import GenerationWorkflow, payload_hashes
from spatial.v2.entailment_thinking_training import tokenize
from spatial.v2.serialization import atomic_json


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        policy.MODEL["model_id"],
        revision=policy.MODEL["revision"],
        local_files_only=True,
    )


@pytest.fixture
def campaign(tmp_path):
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    for name in ("data", "audit", "inputs", "tokenizers"):
        (artifact / name).mkdir()
    cfg = policy.workflow_config()
    manifest = {
        "schema": cfg.manifest_schema,
        "status": "smoke-passed",
        "mode": "smoke",
        "model": policy.MODEL,
        "training_ready": False,
        "pending_release_checks": ["paired-example-review"],
        "source_sha256": GenerationWorkflow(cfg).fingerprints(),
        "payload_sha256": payload_hashes(artifact),
    }
    atomic_json(artifact / "manifest.json", manifest)
    atomic_json(
        artifact / "generation.json",
        {
            "status": "smoke-passed",
            "manifest_sha256": sft.digest(artifact / "manifest.json"),
        },
    )
    output = tmp_path / "campaign"
    sft.prepare(cfg, artifact, output)
    return output


def test_four_configs_share_data_and_match_within_seed(tmp_path):
    from transformers.trainer_pt_utils import AcceleratorConfig

    artifacts = tmp_path / "artifacts"
    configs = {}
    for style in sft.STYLES:
        for seed in sft.SEEDS:
            directory = tmp_path / sft.run_key(style, seed)
            config = sft.config_for(policy.MODEL, artifacts, directory, style, seed)
            assert config["seed"] == config["data_seed"] == seed
            assert AcceleratorConfig(
                **config["accelerator_config"]
            ).use_seedable_sampler
            assert config["eval_strategy"] == config["save_strategy"] == "epoch"
            assert config["val_set_size"] == 0
            assert config["test_datasets"][0]["path"] == str(
                artifacts / f"data/dev/{style}-tokenized.jsonl"
            )
            assert config["num_epochs"] == 2
            assert config["lora_r"] == 64 and config["lora_alpha"] == 128
            assert (
                config["skip_prepare_dataset"]
                and not config["dataset_exact_deduplication"]
            )
            assert config["wandb_project"] == "spatialentail-final-v1-sft"
            configs[style, seed] = config
    for seed in sft.SEEDS:
        normalized = []
        for style in sft.STYLES:
            cfg = deepcopy(configs[style, seed])
            for key in (
                "output_dir",
                "datasets",
                "test_datasets",
                "dataset_prepared_path",
                "wandb_name",
                "wandb_run_id",
            ):
                cfg.pop(key)
            normalized.append(cfg)
        assert normalized[0] == normalized[1]
    for style in sft.STYLES:
        assert configs[style, 42]["datasets"] == configs[style, 43]["datasets"]
    assert len({c["output_dir"] for c in configs.values()}) == 4
    assert len({c["wandb_run_id"] for c in configs.values()}) == 4


def test_wandb_ids_differ_across_submitter_shaped_campaigns(tmp_path):
    identities = []
    for submission in ("sft-FIRST", "sft-SECOND"):
        for style in sft.STYLES:
            for seed in sft.SEEDS:
                directory = (
                    tmp_path / submission / "campaign" / sft.run_key(style, seed)
                )
                cfg = sft.config_for(
                    policy.MODEL, tmp_path / "artifacts", directory, style, seed
                )
                assert cfg["wandb_name"] == cfg["wandb_run_id"]
                assert cfg["wandb_run_id"].endswith(f"-{style}-seed-{seed}")
                assert not any(c in cfg["wandb_run_id"] for c in "/\\#?%:")
                assert cfg == sft.config_for(
                    policy.MODEL, tmp_path / "artifacts", directory, style, seed
                )
                identities.append(cfg["wandb_run_id"])
    assert len(set(identities)) == 8


def test_training_group_matches_config_and_overrides_stale_environment(
    campaign, monkeypatch
):
    captured = []
    monkeypatch.setenv("WANDB_RUN_GROUP", "stale-group")
    monkeypatch.setattr(sft, "verify_environment", lambda: None)
    monkeypatch.setattr(sft, "selected_checkpoint", lambda *_: None)
    monkeypatch.setattr(sft.subprocess, "run", lambda *a, **kw: captured.append(kw))
    verify = sft.verify_run
    monkeypatch.setattr(sft, "verify_run", lambda p: verify(p, require_ready=False))
    for style in sft.STYLES:
        for seed in sft.SEEDS:
            directory = campaign / sft.run_key(style, seed)
            sft.train_run(directory)
            cfg = sft.config_for(
                policy.MODEL, campaign.parent / "artifacts", directory, style, seed
            )
            group = captured[-1]["env"]["WANDB_RUN_GROUP"]
            assert group != "stale-group"
            assert cfg["wandb_run_id"] == f"{group}-{style}-seed-{seed}"
    assert len({call["env"]["WANDB_RUN_GROUP"] for call in captured}) == 1
    assert "WANDB_RUN_GROUP=" not in (EXP / "slurm/train.sh").read_text()


def test_real_axolotl_plugin_forwards_nondefault_trainer_arguments(tmp_path):
    pytest.importorskip("axolotl.integrations.base")
    from transformers import TrainingArguments

    from spatial.v2.entailment_sft_plugin import FinalSFTPlugin

    for seed in sft.SEEDS:
        extra = FinalSFTPlugin().get_training_args(SimpleNamespace(seed=seed))
        args = TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to="none", seed=seed, **extra
        )
        assert args.data_seed == seed
        assert args.prediction_loss_only
        assert not args.logging_nan_inf_filter


@pytest.fixture
def model_audit_input(tmp_path, monkeypatch):
    from transformers import TrainingArguments, TrainerState

    config = sft.config_for(
        policy.MODEL, tmp_path / "artifacts", tmp_path / "run", "natural", 42
    )
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to="none",
        seed=42,
        data_seed=42,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        max_grad_norm=0.05,
        warmup_steps=75,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_only_model=True,
        logging_nan_inf_filter=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
    )
    # Exercise the real arguments API without requiring a CUDA model locally.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    parameters = [
        (
            f"model.layers.0.{target}.lora_{matrix}.default.weight",
            SimpleNamespace(
                requires_grad=True, device=torch.device("cuda"), numel=lambda: 64
            ),
        )
        for target in config["lora_target_modules"]
        for matrix in ("A", "B")
    ]
    model = SimpleNamespace(
        config=SimpleNamespace(
            _commit_hash=policy.MODEL["revision"],
            _attn_implementation="sdpa",
            num_hidden_layers=1,
            quantization_config=SimpleNamespace(
                to_dict=lambda: dict(config["bnb_config_kwargs"])
            ),
        ),
        is_loaded_in_4bit=True,
        peft_config={
            "default": SimpleNamespace(
                r=64,
                lora_alpha=128,
                lora_dropout=0,
                bias="none",
                target_modules=config["lora_target_modules"],
            )
        },
        named_parameters=lambda: iter(parameters),
    )
    trainer = SimpleNamespace(
        args=args, model=model, state=TrainerState(max_steps=1500)
    )
    return trainer, {"seed": 42, "model": policy.MODEL}, config


@pytest.mark.parametrize("warmup", [75, 0.05])
def test_model_audit_accepts_pinned_trainer_arguments(model_audit_input, warmup):
    trainer, meta, config = model_audit_input
    trainer.args.warmup_steps = warmup
    assert not hasattr(trainer.args, "warmup_ratio")
    assert not hasattr(trainer.args, "group_by_length")
    result = runtime.audit_model(trainer, meta, config)
    assert result["trainable_tensors"] == len(config["lora_target_modules"]) * 2


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("warmup_steps", 0, "warmup steps"),
        ("warmup_steps", 74, "warmup steps"),
        ("warmup_steps", 76, "warmup steps"),
        ("warmup_steps", 0.1, "warmup steps"),
        ("train_sampling_strategy", "group_by_length", "train_sampling_strategy"),
        ("train_sampling_strategy", "sequential", "train_sampling_strategy"),
        ("data_seed", 43, "data_seed"),
    ],
)
def test_model_audit_rejects_changed_recipe(model_audit_input, name, value, message):
    trainer, meta, config = model_audit_input
    setattr(trainer.args, name, value)
    with pytest.raises(ValueError, match=message):
        runtime.audit_model(trainer, meta, config)


def test_prepare_does_not_authorize_smoke_or_overwrite(campaign):
    for style in sft.STYLES:
        for seed in sft.SEEDS:
            directory = campaign / sft.run_key(style, seed)
            assert sft.verify_run(directory, require_ready=False)["seed"] == seed
            with pytest.raises(ValueError, match="not a training release"):
                sft.verify_run(directory)
    with pytest.raises(ValueError, match="fresh campaign"):
        sft.prepare(policy.workflow_config(), campaign.parent / "artifacts", campaign)


@pytest.mark.parametrize("name", ("config.yaml", "run.json"))
def test_tampered_campaign_blocks_launch(campaign, name):
    run = campaign / "natural-seed-42"
    with (run / name).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="campaign file changed"):
        sft.verify_run(run, require_ready=False)


def test_failed_release_never_starts_trainer(campaign, monkeypatch):
    invoked = []
    monkeypatch.setattr(sft.subprocess, "run", lambda *a, **k: invoked.append(a))
    with pytest.raises(ValueError, match="not a training release"):
        sft.train_run(campaign / "natural-seed-42")
    assert invoked == []


@pytest.mark.parametrize(
    "correct,expected", [((600, 700), 2), ((700, 600), 1), ((700, 700), 1)]
)
def test_dev_selection_uses_accuracy_and_earliest_tie(correct, expected):
    epochs = [
        {"epoch": i, "correct": c, "total": 1200, "dev_loss": 3 - i}
        for i, c in enumerate(correct, 1)
    ]
    assert sft.choose_epoch(epochs)["epoch"] == expected
    epochs[0]["total"] = 1199
    with pytest.raises(ValueError, match="incomplete"):
        sft.choose_epoch(epochs)


def make_tokens(tokenizer):
    rows = []
    for proof in ("P1 proves North.", "P1 proves North. " * 5):
        source = [
            {"role": "system", "content": "Use entailment and end with Answer: X."},
            {"role": "user", "content": "Gym is north of Bank. Which direction?"},
            {"role": "assistant", "content": proof.strip() + "\n\nAnswer: A"},
        ]
        rows.append(tokenize(tokenizer, source, "A", policy.MODEL)[0])
    return rows


@pytest.mark.parametrize("side", ("left", "right"))
def test_real_collator_preserves_native_masks(tokenizer, side):
    original = tokenizer.padding_side
    tokenizer.padding_side = side
    try:
        dataset = make_tokens(tokenizer)
        collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
        assert (
            runtime.audit_collator(collator, dataset, tokenizer)["status"] == "passed"
        )

        def broken(rows):
            batch = collator(rows)
            batch["labels"][batch["attention_mask"] == 0] = 0
            return batch

        with pytest.raises(ValueError, match="padding"):
            runtime.audit_collator(broken, dataset, tokenizer)
    finally:
        tokenizer.padding_side = original


@pytest.mark.parametrize("mutation", (None, "order", "mask", "count"))
def test_loaded_data_must_match_every_sealed_token(tmp_path, tokenizer, mutation):
    directory = tmp_path / "data/train"
    directory.mkdir(parents=True)
    rows = make_tokens(tokenizer)
    sidecars = []
    for i, row in enumerate(rows):
        import hashlib

        sidecars.append(
            {
                "id": str(i),
                "token_sha256": {
                    "natural": hashlib.sha256(
                        json.dumps(row, sort_keys=True).encode()
                    ).hexdigest(),
                },
            }
        )
    for name, values in (
        ("natural-tokenized", rows),
        ("index", sidecars),
        ("references", [{"id": str(i)} for i in range(2)]),
    ):
        (directory / f"{name}.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in values)
        )
    loaded = deepcopy(rows)
    if mutation == "order":
        loaded.reverse()
    elif mutation == "mask":
        loaded[0]["labels"][0] = 0
    elif mutation == "count":
        loaded.pop()
    if mutation:
        with pytest.raises(ValueError):
            runtime.audit_loaded(loaded, tmp_path, "train", "natural")
    else:
        assert (
            runtime.audit_loaded(loaded, tmp_path, "train", "natural")["examples"] == 2
        )


def test_generated_dev_uses_only_prompts_and_restores_state(tmp_path, tokenizer):
    from spatial.v2.composition_training import messages

    rows = [
        {
            "id": str(i),
            "contract": "unique-entailed",
            "answer": "A",
            "prompt": "Gym is north of Bank.",
            "references": {"natural": "SECRET GOLD"},
        }
        for i in range(2)
    ]
    completion = tokenizer.encode(
        "P1.\n</think>\n\nAnswer: A<|im_end|>", add_special_tokens=False
    )
    seen = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(use_cache=False)

        def generate(self, input_ids, attention_mask, generation_config):
            seen.extend(input_ids.tolist())
            assert generation_config.do_sample is False
            torch.rand(1)
            return torch.cat(
                (input_ids, torch.tensor([completion] * len(input_ids))), dim=1
            )

    model = Model()
    trainer = SimpleNamespace(
        processing_class=tokenizer,
        model=model,
        accelerator=SimpleNamespace(unwrap_model=lambda m: m),
    )
    previous = tokenizer.padding_side
    rng = torch.get_rng_state()
    result = runtime.generated_dev(
        trainer, rows, "natural", policy.MODEL, tmp_path / "predictions.jsonl"
    )
    assert result["correct"] == result["total"] == 2
    assert model.training and not model.config.use_cache
    assert tokenizer.padding_side == previous and torch.equal(
        torch.get_rng_state(), rng
    )
    assert seen == [
        tokenizer.apply_chat_template(
            messages(r, "natural"),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        for r in rows
    ]
    assert all("SECRET GOLD" not in tokenizer.decode(ids) for ids in seen)


def test_numerical_checks_abort_before_optimizer():
    callback = object.__new__(runtime.EpochDevCallback)
    model = torch.nn.Linear(2, 1)
    for p in model.parameters():
        p.grad = torch.full_like(p, float("nan"))
    with pytest.raises(ValueError, match="nonfinite gradients"):
        callback.on_pre_optimizer_step(None, None, None, model=model)
    with pytest.raises(ValueError, match="nonfinite loss"):
        callback.on_log(
            None, SimpleNamespace(global_step=21), None, logs={"loss": float("inf")}
        )


@pytest.mark.parametrize("correct,selected", [((600, 700), 2), ((600, 600), 1)])
def test_epoch_callback_saves_selects_and_reloads_real_adapters(
    tmp_path, monkeypatch, correct, selected
):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    artifacts = tmp_path / "artifacts"
    (artifacts / "data/dev").mkdir(parents=True)
    rows = [
        {"id": str(i), "split": "dev", "contract": "unique-entailed", "answer": "A"}
        for i in range(1200)
    ]
    (artifacts / "data/dev/references.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    directory = tmp_path / "natural-seed-42"
    directory.mkdir()
    meta = {
        "artifacts": str(artifacts),
        "style": "natural",
        "seed": 42,
        "model": policy.MODEL,
        "generation_manifest_sha256": "frozen",
    }
    monkeypatch.setattr(runtime, "verify_run", lambda *_: meta)
    monkeypatch.setattr(sft, "verify_run", lambda *_: meta)
    model = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        ),
        LoraConfig(r=2, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    trainer = SimpleNamespace(
        model=model,
        accelerator=SimpleNamespace(unwrap_model=lambda m: m),
        log=lambda _: None,
    )
    callback = runtime.EpochDevCallback(trainer, directory, meta)
    callback.epoch_tokens = 10
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    args = SimpleNamespace(output_dir=str(directory / "model"))
    for epoch, count in enumerate(correct, 1):
        checkpoint = Path(args.output_dir) / f"checkpoint-{750 * epoch}"
        model.save_pretrained(checkpoint)

        def generate(trainer, dev, style, policy, output, count=count):
            output.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": r["id"],
                            "text": f"Answer: {'A' if i < count else 'B'}",
                        }
                    )
                    + "\n"
                    for i, r in enumerate(dev)
                )
            )
            return {"correct": count, "total": len(dev)}

        monkeypatch.setattr(runtime, "generated_dev", generate)
        state = SimpleNamespace(global_step=750 * epoch, epoch=float(epoch))
        callback.examples = 6000 * epoch
        callback.supervised_tokens = 10 * epoch
        callback.on_evaluate(args, state, None, metrics={"eval_loss": 1 / epoch})
        callback.on_save(args, state, None)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.add_(0.25)
    callback.on_train_end(args, state, None)
    result = json.loads((directory / "selection.json").read_text())
    assert (
        result["epoch"] == selected
        and result["checkpoint_reload"] == "both-epochs-passed"
    )
    assert {n: p.requires_grad for n, p in model.named_parameters()} == flags
    assert list(model.peft_config) == ["default"]
    _, checkpoint, epoch = sft.selected_checkpoint(directory)
    assert epoch == selected and checkpoint.name == f"checkpoint-{750 * selected}"
    with (directory / "dev-epoch-1.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="predictions changed"):
        sft.selected_checkpoint(directory)


@pytest.mark.parametrize("spooled", (False, True))
@pytest.mark.parametrize(
    "script,mode",
    [("train.sh", None), ("evaluate.sh", "selected"), ("evaluate.sh", "untuned")],
)
def test_workers_resolve_repository_after_slurm_copies_script(
    tmp_path, script, mode, spooled
):
    root = tmp_path / "repo with spaces"
    exp = root / "experiments/spatialentail-final-v1"
    (exp / "slurm").mkdir(parents=True)
    (root / "slurm/lib").mkdir(parents=True)
    (root / "slurm/lib/pin-srun-cpus.sh").write_text(
        (ROOT / "slurm/lib/pin-srun-cpus.sh").read_text()
    )
    common = root / "experiments/15-v2-ablation/jobs/common.bash"
    common.parent.mkdir(parents=True)
    common.write_text("load_wandb_credentials() { export CREDENTIALS_LOADED=1; }\n")
    (exp / "sft.py").write_text("")
    worker = (
        tmp_path / "var/lib/slurm/slurmd/job1064849/slurm_script"
        if spooled
        else exp / "slurm" / script
    )
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker.write_text((EXP / "slurm" / script).read_text())
    bins = tmp_path / "bin"
    bins.mkdir()
    for name, body in (
        ("srun", '#!/bin/bash\nexec "$@"\n'),
        ("nvidia-smi", "#!/bin/bash\nexit 0\n"),
        (
            "uv",
            (
                '#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n'
                'printf "%s\\n" "$PYTHONPATH" "$SRUN_CPUS_PER_TASK" '
                '"${CREDENTIALS_LOADED:-unset}" > "$ENV_CAPTURE"\n'
            ),
        ),
    ):
        path = bins / name
        path.write_text(body)
        path.chmod(0o755)
    run = tmp_path / "campaign elsewhere/natural-seed-42"
    run.mkdir(parents=True)
    capture, environment = tmp_path / "args", tmp_path / "environment"
    arguments = [str(run)] if mode is None else [str(run), "sft-bench", mode]
    result = subprocess.run(
        ["bash", str(worker), *arguments],
        cwd=root,
        env={
            **os.environ,
            "PATH": f"{bins}:{os.environ['PATH']}",
            "PYTHONPATH": "existing-path",
            "CAPTURE": str(capture),
            "ENV_CAPTURE": str(environment),
            "SLURM_JOB_ID": "1064849",
            "SLURM_SUBMIT_DIR": str(tmp_path),
            "SLURM_TRES_PER_TASK": "cpu=8",
            "SLURM_CPUS_PER_TASK": "64",
            "SLURM_SRUN_CPUS_PINNED": "",
            "CREDENTIALS_LOADED": "",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    resolved = root.resolve()
    expected = [
        "run",
        "--project",
        str(resolved),
        "--frozen",
        "--no-sync",
        "python",
        str(resolved / "experiments/spatialentail-final-v1/sft.py"),
        "train" if mode is None else "evaluate",
        "--run",
        str(run),
    ]
    if mode is not None:
        expected += ["--benchmark", "sft-bench"]
    if mode == "untuned":
        expected += ["--untuned"]
    assert capture.read_text().splitlines() == expected
    assert environment.read_text().splitlines() == [
        f"{resolved}:existing-path",
        "8",
        "1" if mode is None else "unset",
    ]


@pytest.mark.parametrize(
    "script,count", [("submit-training.sh", 4), ("submit-evaluation.sh", 12)]
)
def test_submitters_count_jobs_and_stop_on_failed_verification(tmp_path, script, count):
    root = tmp_path / "repo with spaces"
    exp = root / "experiments/spatialentail-final-v1"
    (exp / "slurm").mkdir(parents=True)
    (exp / "slurm" / script).write_text((EXP / "slurm" / script).read_text())
    argument = tmp_path / "artifacts or campaign"
    argument.mkdir()
    receipt = tmp_path / "release.json"
    receipt.write_text("{}\n")
    arguments = (
        [str(argument), str(receipt)]
        if script == "submit-training.sh"
        else [str(argument)]
    )
    bins = tmp_path / "bin"
    bins.mkdir()
    calls = tmp_path / "calls"
    (bins / "uv").write_text('#!/bin/bash\n[[ "${FAIL_VERIFY:-}" != 1 ]]\n')
    (bins / "sbatch").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\necho "12345;cluster"\n'
    )
    for file in bins.iterdir():
        file.chmod(0o755)
    env = {**os.environ, "PATH": f"{bins}:{os.environ['PATH']}", "CALLS": str(calls)}
    subprocess.run(
        ["bash", str(exp / "slurm" / script), *arguments], env=env, check=True
    )
    submitted = calls.read_text().splitlines()
    assert len(submitted) == count
    assert all(f"--chdir={root}" in line for line in submitted)
    if script == "submit-training.sh":
        assert [line.rsplit("/", 1)[-1] for line in submitted] == [
            "natural-seed-42",
            "symbolic-seed-42",
            "natural-seed-43",
            "symbolic-seed-43",
        ]
        assert all("--dependency" not in line for line in submitted)
    calls.unlink()
    result = subprocess.run(
        ["bash", str(exp / "slurm" / script), *arguments],
        env={**env, "FAIL_VERIFY": "1"},
        check=False,
    )
    assert result.returncode != 0 and not calls.exists()
