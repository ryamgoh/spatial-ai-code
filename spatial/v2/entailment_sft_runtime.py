"""Trainer-boundary checks and in-training generated Dev evaluation."""

import hashlib
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from transformers import GenerationConfig, TrainerCallback

from spatial.v2.composition_evaluation import adapter_hash, parse_or_none
from spatial.v2.composition_training import messages
from spatial.v2.entailment_generation import digest, read_jsonl
from spatial.v2.entailment_sft import choose_epoch, config_for, verify_run
from spatial.v2.entailment_thinking_training import require
from spatial.v2.serialization import atomic_json

TOKEN_FIELDS = ("input_ids", "attention_mask", "labels")


def token_record(row):
    return {
        key: row[key].tolist() if hasattr(row[key], "tolist") else list(row[key])
        for key in TOKEN_FIELDS
    }


def audit_loaded(dataset, artifacts, split, style):
    expected = read_jsonl(artifacts / f"data/{split}/{style}-tokenized.jsonl")
    sidecar = read_jsonl(artifacts / f"data/{split}/index.jsonl")
    refs = read_jsonl(artifacts / f"data/{split}/references.jsonl")
    require(
        dataset is not None
        and len(dataset) == len(expected) == len(sidecar) == len(refs),
        f"{split}: loader changed row count",
    )
    require(
        [r["id"] for r in refs] == [r["id"] for r in sidecar],
        f"{split}: sidecar IDs differ",
    )
    target_tokens = 0
    for loaded, sealed, index in zip(dataset, expected, sidecar, strict=True):
        actual = token_record(loaded)
        require(
            actual == sealed,
            f"{split}: loaded tokens/masks/order changed: {index['id']}",
        )
        require(
            hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
            == index["token_sha256"][style],
            f"{split}: token hash mismatch",
        )
        target_tokens += sum(label != -100 for label in actual["labels"])
    return {"examples": len(expected), "supervised_tokens": target_tokens}


def audit_collator(collator, dataset, tokenizer):
    lengths = [len(row["input_ids"]) for row in dataset]
    indices = [lengths.index(min(lengths)), lengths.index(max(lengths))]
    require(min(lengths) < max(lengths), "mixed-length collator probe required")
    rows = [token_record(dataset[i]) for i in indices]
    batch = collator(deepcopy(rows))
    for index, row in enumerate(rows):
        mask = batch["attention_mask"][index].tolist()
        ids = batch["input_ids"][index].tolist()
        labels = batch["labels"][index].tolist()
        size = len(row["input_ids"])
        require(
            len(mask) == len(ids) == len(labels) and len(mask) >= size,
            "collator array lengths differ",
        )
        padding = len(mask) - size
        expected_mask = (
            [0] * padding + [1] * size
            if tokenizer.padding_side == "left"
            else [1] * size + [0] * padding
        )
        require(mask == expected_mask, "collator attention mask changed")
        require(
            [v for v, keep in zip(ids, mask) if keep] == row["input_ids"]
            and [v for v, keep in zip(labels, mask) if keep] == row["labels"],
            "collator changed unpadded tokens/labels",
        )
        require(
            all(
                label == -100 and token == tokenizer.pad_token_id
                for label, token, keep in zip(labels, ids, mask)
                if not keep
            ),
            "collator padding is supervised or uses wrong token",
        )
    return {"mixed_lengths": [min(lengths), max(lengths)], "status": "passed"}


def audit_model(trainer, meta, config):
    args, model = trainer.args, trainer.model
    require(
        args.world_size == 1
        and torch.cuda.is_available()
        and torch.cuda.device_count() == 1,
        "one visible CUDA GPU required",
    )
    require(
        model.config._commit_hash == meta["model"]["revision"], "model revision differs"
    )
    require(model.config._attn_implementation == "sdpa", "SDPA required")
    require(getattr(model, "is_loaded_in_4bit", False), "QLoRA base is not 4-bit")
    quant = model.config.quantization_config.to_dict()
    for name, expected in config["bnb_config_kwargs"].items():
        require(quant[name] == expected, f"quantization setting differs: {name}")
    lora = model.peft_config["default"]
    require(
        lora.r == 64
        and lora.lora_alpha == 128
        and lora.lora_dropout == 0
        and lora.bias == "none"
        and set(lora.target_modules) == set(config["lora_target_modules"]),
        "LoRA configuration differs",
    )
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    require(trainable, "no trainable adapter parameters")
    require(
        all(
            (".lora_A." in n or ".lora_B." in n) and p.device.type == "cuda"
            for n, p in trainable.items()
        ),
        "unexpected trainable/base parameters",
    )
    targets = {n.split(".lora_")[0].rsplit(".", 1)[-1] for n in trainable}
    require(targets == set(config["lora_target_modules"]), "missing LoRA targets")
    require(
        len(trainable) == model.config.num_hidden_layers * len(targets) * 2,
        "LoRA targets do not cover every layer",
    )
    expected_args = {
        "seed": meta["seed"],
        "data_seed": meta["seed"],
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "per_device_eval_batch_size": 1,
        "learning_rate": 5e-5,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 0.05,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "load_best_model_at_end": False,
        "save_only_model": True,
        "logging_nan_inf_filter": False,
        "train_sampling_strategy": "random",
    }
    for name, expected in expected_args.items():
        require(
            getattr(args, name) == expected,
            f"effective trainer setting differs: {name}",
        )
    # Axolotl resolves its YAML warmup ratio into Transformers v5 warmup_steps.
    require(
        args.get_warmup_steps(trainer.state.max_steps)
        == math.ceil(trainer.state.max_steps * config["warmup_ratio"]),
        "effective trainer setting differs: warmup steps",
    )
    for name, expected in (
        ("optim", "paged_adamw_8bit"),
        ("lr_scheduler_type", "cosine"),
        ("eval_strategy", "epoch"),
        ("save_strategy", "epoch"),
    ):
        value = getattr(args, name)
        require(getattr(value, "value", value) == expected, f"wrong {name}")
    return {
        "quantization": quant,
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "trainable_tensors": len(trainable),
        "targets": sorted(targets),
        "training_arguments": args.to_dict(),
    }


def generated_dev(trainer, rows, style, policy, output, batch_size=4):
    """Decode raw Stage-1 tokens without reference answers in model inputs."""
    tokenizer = trainer.processing_class
    model = trainer.accelerator.unwrap_model(trainer.model)
    device = next(model.parameters()).device
    training, padding, cache = (
        model.training,
        tokenizer.padding_side,
        model.config.use_cache,
    )
    python_state, numpy_state = random.getstate(), np.random.get_state()
    correct = valid = total_tokens = exhausted = 0
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        model.eval()
        tokenizer.padding_side = "left"
        with (
            torch.random.fork_rng(devices=devices),
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
            output.open("x") as stream,
        ):
            for offset in range(0, len(rows), batch_size):
                batch_rows = rows[offset : offset + batch_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        messages(r, style),
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                    )
                    for r in batch_rows
                ]
                require(
                    max(map(len, prompts)) + policy["stage1_max_new_tokens"]
                    <= policy["context_length"],
                    "Dev generation would overflow",
                )
                batch = tokenizer.pad(
                    {
                        "input_ids": prompts,
                        "attention_mask": [[1] * len(p) for p in prompts],
                    },
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                width = batch["input_ids"].shape[1]
                generated = model.generate(
                    **batch,
                    generation_config=GenerationConfig(
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=policy["stage1_max_new_tokens"],
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    ),
                )
                require(
                    len(generated) == len(batch_rows), "Dev generation count differs"
                )
                for row, sequence in zip(batch_rows, generated, strict=True):
                    ids = sequence[width:].tolist()
                    if tokenizer.eos_token_id in ids:
                        ids = ids[: ids.index(tokenizer.eos_token_id) + 1]
                    raw = tokenizer.decode(ids, skip_special_tokens=False)
                    text = raw.split("</think>", 1)[1] if "</think>" in raw else ""
                    text = text.removesuffix("<|im_end|>").strip()
                    answer = parse_or_none(text, row["contract"], 1)
                    correct += answer == row["answer"]
                    valid += answer is not None
                    total_tokens += len(ids)
                    exhausted += (
                        len(ids) == policy["stage1_max_new_tokens"]
                        and ids[-1] != tokenizer.eos_token_id
                    )
                    stream.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "text": text,
                                "raw_text": raw,
                                "token_ids": ids,
                                "correct": answer == row["answer"],
                            }
                        )
                        + "\n"
                    )
                stream.flush()
                print(
                    f"Dev Stage 1: {min(offset + batch_size, len(rows))}/{len(rows)}",
                    flush=True,
                )
    finally:
        tokenizer.padding_side = padding
        model.config.use_cache = cache
        model.train(training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return {
        "correct": correct,
        "total": len(rows),
        "format_valid": valid,
        "generated_tokens": total_tokens,
        "budget_exhausted": exhausted,
    }


class EpochDevCallback(TrainerCallback):
    def __init__(self, trainer, directory, meta):
        self.trainer, self.directory, self.meta = trainer, directory, meta
        self.artifacts = Path(meta["artifacts"])
        self.dev = read_jsonl(self.artifacts / "data/dev/references.jsonl")
        require(
            len(self.dev) == 1200 and all(r["split"] == "dev" for r in self.dev),
            "fixed Dev set required",
        )
        self.epochs, self.losses = [], {}
        self.examples = self.supervised_tokens = 0
        self.hook = None

    def on_train_begin(self, args, state, control, **kwargs):
        require(
            state.global_step == 0 and state.max_steps == 1500,
            "fresh 1,500-step run required",
        )
        verify_run(self.directory)
        tokenizer = self.trainer.processing_class
        config = config_for(
            self.meta["model"],
            self.artifacts,
            self.directory,
            self.meta["style"],
            self.meta["seed"],
        )
        audits = {
            split: audit_loaded(dataset, self.artifacts, split, self.meta["style"])
            for split, dataset in (
                ("train", self.trainer.train_dataset),
                ("dev", self.trainer.eval_dataset),
            )
        }
        self.epoch_tokens = audits["train"]["supervised_tokens"]
        audits["collator"] = audit_collator(
            self.trainer.data_collator, self.trainer.train_dataset, tokenizer
        )
        audits["model"] = audit_model(self.trainer, self.meta, config)
        require(
            len(kwargs["train_dataloader"]) == 6000,
            "loader must yield 6,000 microbatches",
        )
        import wandb

        require(
            wandb.run is not None and wandb.run.settings.mode == "online",
            "online W&B run required",
        )
        audits["wandb_run_id"] = wandb.run.id
        atomic_json(self.directory / "trainer-boundary.json", audits)

        def check_forward(module, inputs, named, output):
            loss = getattr(output, "loss", None)
            if loss is not None:
                require(
                    bool(torch.isfinite(loss.detach()).all()), "nonfinite model loss"
                )
            if module.training and "labels" in named:
                self.examples += named["labels"].shape[0]
                self.supervised_tokens += int((named["labels"] != -100).sum())

        self.hook = self.trainer.model.register_forward_hook(
            check_forward, with_kwargs=True
        )

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        grads = [
            p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        require(
            grads and bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()),
            "missing or nonfinite gradients",
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        for name in ("loss", "grad_norm", "eval_loss"):
            if name in logs:
                require(math.isfinite(logs[name]), f"nonfinite {name}")
        if state.global_step <= 20 and "loss" in logs:
            with (self.directory / "opening-steps.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": state.global_step,
                            **logs,
                            "examples_seen": self.examples,
                            "supervised_tokens_seen": self.supervised_tokens,
                            "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
                        }
                    )
                    + "\n"
                )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        require(
            metrics and math.isfinite(metrics["eval_loss"]), "finite Dev loss required"
        )
        self.losses[state.global_step] = metrics["eval_loss"]

    def on_save(self, args, state, control, **kwargs):
        epoch = round(state.epoch)
        require(
            epoch in (1, 2)
            and state.global_step == epoch * 750
            and abs(state.epoch - epoch) < 1e-6,
            "unexpected epoch checkpoint",
        )
        require(
            state.global_step in self.losses, "Dev loss was not evaluated this epoch"
        )
        require(
            self.examples == 6000 * epoch
            and self.supervised_tokens == self.epoch_tokens * epoch,
            "training exposure differs from the sealed dataset",
        )
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        identity = adapter_hash(checkpoint)
        predictions = self.directory / f"dev-epoch-{epoch}.jsonl"
        metrics = generated_dev(
            self.trainer, self.dev, self.meta["style"], self.meta["model"], predictions
        )
        record = {
            **metrics,
            "epoch": epoch,
            "step": state.global_step,
            "dev_loss": self.losses[state.global_step],
            "adapter_sha256": identity,
            "predictions_sha256": digest(predictions),
            "dev_sha256": digest(self.artifacts / "data/dev/references.jsonl"),
        }
        atomic_json(self.directory / f"dev-epoch-{epoch}.json", record)
        self.epochs.append(record)
        self.trainer.log({"dev_stage1_accuracy": metrics["correct"] / metrics["total"]})

    def on_train_end(self, args, state, control, **kwargs):
        if self.hook is not None:
            self.hook.remove()
        require(state.global_step == 1500, "training did not complete both epochs")
        best = choose_epoch(self.epochs)
        verify_run(self.directory)
        from peft import get_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights

        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        # Exercise PEFT's real adapter loader, retaining the original active adapter.
        original = model.active_adapter
        flags = {name: p.requires_grad for name, p in model.named_parameters()}
        for epoch in self.epochs:
            checkpoint = Path(args.output_dir) / f"checkpoint-{epoch['step']}"
            require(
                adapter_hash(checkpoint) == epoch["adapter_sha256"],
                "checkpoint changed",
            )
            try:
                model.load_adapter(
                    str(checkpoint), adapter_name="reload_check", is_trainable=False
                )
                loaded = get_peft_model_state_dict(model, adapter_name="reload_check")
                saved = load_peft_weights(str(checkpoint), device="cpu")
                require(
                    set(loaded) == set(saved)
                    and all(
                        torch.equal(loaded[k].detach().cpu(), saved[k]) for k in saved
                    ),
                    "checkpoint reload differs from saved adapter",
                )
            finally:
                if "reload_check" in model.peft_config:
                    model.delete_adapter("reload_check")
                model.set_adapter(original)
                for name, param in model.named_parameters():
                    if name in flags:
                        param.requires_grad_(flags[name])
        atomic_json(
            self.directory / "selection.json",
            {
                "style": self.meta["style"],
                "seed": self.meta["seed"],
                "generation_manifest_sha256": self.meta["generation_manifest_sha256"],
                "epoch": best["epoch"],
                "adapter_sha256": best["adapter_sha256"],
                "criterion": "generated-stage1-dev-accuracy; earliest epoch on ties",
                "epoch_receipts_sha256": {
                    str(e["epoch"]): digest(
                        self.directory / f"dev-epoch-{e['epoch']}.json"
                    )
                    for e in self.epochs
                },
                "checkpoint_reload": "both-epochs-passed",
            },
        )
