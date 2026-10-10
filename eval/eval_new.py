"""
Unified eval entry point: one or two vLLM passes per question, togglable
via CLI (LM Eval Harness).

--stages 1  Single pass: the task scores the complete generated response.
--stages 2  (default) Two passes: free thinking, then a constrained
            A/B/C/D re-ask using the stage-1 output for more reliable
            answers and evaluation.

All existing configs declare `model: vllm_staged_pass`; run them with this
script unchanged (default --stages 2) or pass --stages 1 to run the
single-pass protocol.

Usage:
    cd eval && uv run python eval_new.py --config ../experiments/00-baseline-model/eval-baseline.yaml
    cd eval && uv run python eval_new.py --config <config>.yaml --stages 1
"""

import json
from pathlib import Path

import lm_eval
import typer
from generation_contract import (
    finish_answer,
    prepare_requests,
    reasoning_without_answer,
)
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.config.evaluate_config import EvaluatorConfig
from utils import generate_datetime_id
from vllm.sampling_params import StructuredOutputsParams


@register_model("vllm_staged_pass")
class VLLMStagedPass(LM):
    def __init__(
        self,
        pretrained: str,
        stages: int = 2,
        choices: list[str] | None = None,
        max_thinking_tokens: int = 512,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 8192,
        add_special_tokens: bool = True,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        lora_path: str | None = None,
        max_lora_rank: int = 64,
        **kwargs,
    ):
        super().__init__()
        if stages not in (1, 2):
            raise ValueError(f"stages must be 1 or 2, got {stages}")
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.model_path = pretrained
        self.stages = stages
        self.max_thinking_tokens = max_thinking_tokens
        self.max_model_len = max_model_len
        self.add_special_tokens = add_special_tokens
        self.choices = choices if choices is not None else ["A", "B", "C", "D"]
        self.lora_path = lora_path
        self.LoRARequest = LoRARequest

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, revision=tokenizer_revision or revision
        )

        self.llm = LLM(
            model=pretrained,
            revision=revision,
            tokenizer_revision=tokenizer_revision or revision,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,
            enable_lora=bool(lora_path),
            max_lora_rank=max_lora_rank,
        )

        self.SamplingParams = SamplingParams
        self.StructuredOutputsParams = StructuredOutputsParams

    @property
    def tokenizer_name(self) -> str:
        return self.model_path

    def _get_lora_request(self):
        if not self.lora_path:
            return None
        return self.LoRARequest(
            lora_name="adapter", lora_int_id=1, lora_path=self.lora_path
        )

    def generate_until(self, requests):
        requests = [request.args for request in requests]
        if not requests:
            return []
        prompts = [prompt for prompt, _ in requests]
        token_prompts, options = prepare_requests(
            requests,
            self.tokenizer,
            context_limit=self.max_model_len,
            default_max_tokens=self.max_thinking_tokens,
            add_special_tokens=self.add_special_tokens,
        )
        sampling_params = [self.SamplingParams(**item) for item in options]
        lora_req = self._get_lora_request()
        outputs1 = self.llm.generate(
            token_prompts, sampling_params, lora_request=lora_req
        )

        thinking_outputs = [o.outputs[0].text for o in outputs1]

        if self.stages == 1:
            return thinking_outputs

        reasoning = [reasoning_without_answer(text) for text in thinking_outputs]
        answer_options = {
            "max_gen_toks": 16,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "structured_outputs": self.StructuredOutputsParams(
                regex=f"[{''.join(self.choices)}](,[{''.join(self.choices)}])*"
            ),
        }
        answer_requests = [
            (f"{prompt}\n{text}\n\nAnswer: ", answer_options)
            for prompt, text in zip(prompts, reasoning, strict=True)
        ]
        answer_prompts, options = prepare_requests(
            answer_requests,
            self.tokenizer,
            context_limit=self.max_model_len,
            default_max_tokens=16,
            add_special_tokens=self.add_special_tokens,
        )
        outputs2 = self.llm.generate(
            answer_prompts,
            [self.SamplingParams(**item) for item in options],
            lora_request=None,
        )
        return [
            finish_answer(text, output.outputs[0].text, self.choices)
            for text, output in zip(reasoning, outputs2, strict=True)
        ]

    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not supported for staged evaluation")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "loglikelihood_rolling not supported for staged evaluation"
        )

    def apply_chat_template(self, chat_history: list[dict], **kwargs) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, **kwargs
        )


def run_evaluation(
    config_path: Path,
    stages: int = 2,
    output_dir: Path | None = None,
) -> dict:
    # Results live under the experiment dir that owns the config
    # (e.g. experiments/05-grpo/results/<timestamp>/), unless --output-dir
    # is set (parallel jobs write straight to results/<tag>/).
    config_path = Path(config_path)
    if output_dir is None:
        output_dir = config_path.parent / "results" / generate_datetime_id()
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Newer lm-eval requires output_path whenever log_samples is set; we
    # never pass an EvaluationTracker, so this just satisfies validation.
    yaml_config = EvaluatorConfig.load_yaml_config(config_path)
    yaml_config.setdefault("output_path", str(output_dir))
    bootstrap_iters = yaml_config.pop("bootstrap_iters", 100000)
    if type(bootstrap_iters) is not int or bootstrap_iters < 0:
        raise ValueError("bootstrap_iters must be a nonnegative integer")
    config = EvaluatorConfig(**yaml_config)._configure()
    task_manager = config.process_tasks()

    # Convert structured_outputs dict to StructuredOutputsParams object
    gen_kwargs = config.gen_kwargs.copy() if config.gen_kwargs else {}
    structured_outputs_config = gen_kwargs.pop("structured_outputs", None)

    if structured_outputs_config:
        gen_kwargs["structured_outputs"] = StructuredOutputsParams(
            **structured_outputs_config
        )

    model_args = dict(config.model_args)
    # Only vllm_staged_pass consumes `stages`; other backends reject it.
    if config.model == "vllm_staged_pass":
        model_args["stages"] = stages

    results = lm_eval.simple_evaluate(
        model=config.model,
        model_args=model_args,
        tasks=config.tasks,
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        device=config.device,
        limit=config.limit,
        task_manager=task_manager,
        log_samples=config.log_samples,
        gen_kwargs=gen_kwargs,
        apply_chat_template=config.apply_chat_template,
        system_instruction=config.system_instruction,
        bootstrap_iters=bootstrap_iters,
    )

    if results is not None:
        # Newer lm-eval returns a plain dict; older versions an EvalResults object.
        if isinstance(results, dict):
            results_dict = dict(results)
            samples = results_dict.get("samples") or {}
        else:
            results_dict = getattr(results, "results", results)
            samples = getattr(results, "samples", None) or {}

        # Samples get their own files; keep results.json lean.
        results_dict.pop("samples", None)

        with open(output_dir / "results.json", "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

        if samples:
            for task_name, task_samples in samples.items():
                with open(output_dir / f"responses_{task_name}.jsonl", "w") as f:
                    f.writelines(
                        json.dumps(sample, default=str) + "\n"
                        for sample in task_samples
                    )

        print(f"Results saved to: {output_dir}")

    return results


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = typer.Option(
        ..., "--config", help="Path to the evaluation configuration YAML file"
    ),
    stages: int = typer.Option(
        2,
        "--stages",
        min=1,
        max=2,
        help="1 = single pass; 2 (default) = thinking + constrained A/B/C/D re-ask",
    ),
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help="Write results.json here instead of experiments/<exp>/results/<timestamp>/",
    ),
) -> None:
    """Evaluate a model with one or two vLLM passes."""
    print(f"Running with {stages} stage(s)")
    run_evaluation(Path(config), stages=stages, output_dir=output_dir)
    print(f"\nDone! Config: {config}")


if __name__ == "__main__":
    app()
