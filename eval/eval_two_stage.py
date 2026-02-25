"""
Two-stage evaluation for thinking models using LM Evaluation Harness with vLLM.

Stage 1: Free thinking
Stage 2: Constrained answer (A/B/C/D)

Usage:
    cd eval && uv run python eval_two_stage.py --config ../configs/evals/eval_deepseek-r1-distill-qwen-1.5b_two_pass.yaml
"""

import argparse
import json
from pathlib import Path
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.config.evaluate_config import EvaluatorConfig
import lm_eval
from utils import generate_datetime_id

RESULTS_DIR = Path(__file__).parent.parent / "results"


@register_model("vllm_two_pass")
class VLLMTwoPass(LM):
    def __init__(
        self,
        pretrained: str,
        choices: list[str] | None = None,
        max_thinking_tokens: int = 512,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 8192,
        lora_path: str | None = None,
        **kwargs,
    ):
        super().__init__()
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        self.model_path = pretrained
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

    @property
    def tokenizer_name(self) -> str:
        return self.model_path

    def generate_until(self, requests):
        results = []

        prompts_stage1 = []
        prompts_stage2 = []
        gen_kwargs_list = []

        for request in requests:
            prompt, gen_kwargs = request.args
            gen_kwargs_list.append(gen_kwargs)
            prompts_stage1.append(prompt)

        max_tokens = (
            gen_kwargs_list[0].get("max_gen_toks", self.max_thinking_tokens)
            if gen_kwargs_list
            else self.max_thinking_tokens
        )

        params1 = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            lora_request=self._get_lora_request(),
        )
        outputs1 = self.llm.generate(prompts_stage1, params1)

        thinking_outputs = [o.outputs[0].text for o in outputs1]

        for i, (prompt, thinking) in enumerate(zip(prompts_stage1, thinking_outputs)):
            extraction_prompt = f"{prompt}\n{thinking}\n\nTherefore, the correct option is (just output the letter):"
            prompts_stage2.append(extraction_prompt)

        params2 = self.SamplingParams(
            max_tokens=1,
            temperature=0.0,
            lora_request=self._get_lora_request(),
            structured_outputs=self.StructuredOutputsParams(
                regex="[" + "".join(self.choices) + "]"
            ),
        )
        outputs2 = self.llm.generate(prompts_stage2, params2)

        results = []
        for thinking, o in zip(thinking_outputs, outputs2):
            answer = o.outputs[0].text
            results.append(f"{thinking}\n\nAnswer: {answer}")

        return results

    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not supported for two-pass evaluation")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "loglikelihood_rolling not supported for two-pass evaluation"
        )

    def apply_chat_template(self, chat_history: list[dict], **kwargs) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, **kwargs
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage evaluation with vLLM")
    parser.add_argument(
        "--config", required=True, help="Path to the evaluation configuration YAML file"
    )
    return parser.parse_args()


def run_evaluation(config_path: Path) -> dict:
    config = EvaluatorConfig.from_config(config_path)
    task_manager = config.process_tasks()
    output_dir = RESULTS_DIR / generate_datetime_id()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = lm_eval.simple_evaluate(
        model=config.model,
        model_args=config.model_args,
        tasks=config.tasks,
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        device=config.device,
        limit=config.limit,
        task_manager=task_manager,
        log_samples=config.log_samples,
        gen_kwargs=config.gen_kwargs,
        apply_chat_template=config.apply_chat_template,
        system_instruction=config.system_instruction,
    )

    if results is not None:
        if hasattr(results, "results"):
            results_dict = results.results
        else:
            results_dict = results

        with open(output_dir / "results.json", "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

        if hasattr(results, "samples") and results.samples:
            for task_name, samples in results.samples.items():
                with open(output_dir / f"responses_{task_name}.jsonl", "w") as f:
                    for sample in samples:
                        f.write(json.dumps(sample, default=str) + "\n")

        print(f"Results saved to: {output_dir}")

    return results


def main():
    args = parse_args()
    run_evaluation(Path(args.config))
    print(f"\nDone! Config: {args.config}")


if __name__ == "__main__":
    main()
