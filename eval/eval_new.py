"""
Unified eval entry point: one or two vLLM passes per question, togglable
via CLI (LM Eval Harness).

--stages 1  Single pass: free thinking, answer extracted from the same
            output by the task's regex filter.
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
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.config.evaluate_config import EvaluatorConfig
import lm_eval
from vllm.sampling_params import StructuredOutputsParams
from utils import generate_datetime_id
import typer


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
        lora_path: str | None = None,
        max_lora_rank: int = 64,
        **kwargs,
    ):
        super().__init__()
        if stages not in (1, 2):
            raise ValueError(f"stages must be 1 or 2, got {stages}")
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        self.model_path = pretrained
        self.stages = stages
        self.max_thinking_tokens = max_thinking_tokens
        self.choices = choices if choices is not None else ["A", "B", "C", "D"]
        self.lora_path = lora_path
        self.LoRARequest = LoRARequest

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

        llm_kwargs = dict(
            model=pretrained,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,
            enable_lora=bool(lora_path),
            max_lora_rank=max_lora_rank,
        )

        self.llm = LLM(**llm_kwargs)
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
        prompts = []
        gen_kwargs_list = []

        for request in requests:
            prompt, gen_kwargs = request.args
            gen_kwargs_list.append(gen_kwargs)
            prompts.append(prompt)

        max_tokens = (
            gen_kwargs_list[0].get("max_gen_toks", self.max_thinking_tokens)
            if gen_kwargs_list
            else self.max_thinking_tokens
        )

        params1 = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.6,  # Add some randomness for variety
            repetition_penalty=1.1,  # But still prevent loops
        )
        lora_req = self._get_lora_request()
        outputs1 = self.llm.generate(prompts, params1, lora_request=lora_req)

        thinking_outputs = [o.outputs[0].text for o in outputs1]

        if self.stages == 1:
            # Single pass: answer must be extracted from the thinking
            # output by the task's regex filter.
            return thinking_outputs

        prompts_stage2 = [
            f"{prompt}\n{thinking}\n\nAnswer: "
            for prompt, thinking in zip(prompts, thinking_outputs)
        ]

        params2 = self.SamplingParams(
            max_tokens=16,  # Enough for "A, B, C, D" + punctuation
            temperature=0.0,
            # This regex ensures the model follows the "A, B" format precisely
            structured_outputs=self.StructuredOutputsParams(
                regex=f"[{''.join(self.choices)}](,[{''.join(self.choices)}])*"
            ),
        )

        outputs2 = self.llm.generate(prompts_stage2, params2, lora_request=None)

        final_results = []
        for thinking, o in zip(thinking_outputs, outputs2):
            # o.outputs[0].text will now contain strings like "A" or "A, B"
            answer = o.outputs[0].text
            # We return the full string; your YAML regex filter will then
            # split "A, B" into the list ['A', 'B'] for the metric.
            final_results.append(f"{thinking}\n\nAnswer: {answer}")

        return final_results

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


def run_evaluation(config_path: Path, stages: int = 2) -> dict:
    # Results live under the experiment dir that owns the config
    # (e.g. experiments/05-grpo/results/<timestamp>/).
    config_path = Path(config_path)
    output_dir = config_path.parent / "results" / generate_datetime_id()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Newer lm-eval requires output_path whenever log_samples is set; we
    # never pass an EvaluationTracker, so this just satisfies validation.
    yaml_config = EvaluatorConfig.load_yaml_config(config_path)
    yaml_config.setdefault("output_path", str(output_dir))
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
                    for sample in task_samples:
                        f.write(json.dumps(sample, default=str) + "\n")

        print(f"Results saved to: {output_dir}")

    return results


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = typer.Option(..., "--config", help="Path to the evaluation configuration YAML file"),
    stages: int = typer.Option(2, "--stages", min=1, max=2, help="1 = single pass; 2 (default) = thinking + constrained A/B/C/D re-ask"),
) -> None:
    """Evaluate a model with one or two vLLM passes."""
    print(f"Running with {stages} stage(s)")
    run_evaluation(Path(config), stages=stages)
    print(f"\nDone! Config: {config}")


if __name__ == "__main__":
    app()
