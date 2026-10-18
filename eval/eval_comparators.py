"""Cross-family two-stage evaluation with native model chat templates."""

from __future__ import annotations

from pathlib import Path

import typer
from eval_new import run_evaluation
from generation_contract import (
    answer_regex,
    finish_answer,
    prepare_requests,
    reasoning_without_answer,
)
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from spatial.v2.evaluation_chat import render_chat_input


@register_model("vllm_staged_chat_pass")
class VLLMStagedChatPass(LM):
    """Two-pass vLLM model that renders frozen messages per checkpoint."""

    def __init__(
        self,
        pretrained: str,
        *,
        revision: str,
        tokenizer_revision: str | None = None,
        stages: int = 2,
        choices: list[str] | None = None,
        allow_multiple_answers: bool = True,
        record_stage1: bool = True,
        max_thinking_tokens: int = 4096,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        batch_size: int | None = None,
        chat_template_kwargs: dict | None = None,
        system_message_policy: str = "native",
        trust_remote_code: bool = False,
        **_: object,
    ) -> None:
        del batch_size
        super().__init__()
        if stages not in (1, 2):
            raise ValueError(f"stages must be 1 or 2, got {stages}")
        if not revision:
            raise ValueError("cross-family evaluation requires a pinned revision")

        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self.model_path = pretrained
        self.stages = stages
        self.max_thinking_tokens = max_thinking_tokens
        self.max_model_len = max_model_len
        self.choices = choices or list("ABCDE")
        self.answer_regex = answer_regex(
            self.choices, allow_multiple_answers=allow_multiple_answers
        )
        self.record_stage1 = record_stage1
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.system_message_policy = system_message_policy
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            revision=tokenizer_revision or revision,
            trust_remote_code=trust_remote_code,
        )
        self.llm = LLM(
            model=pretrained,
            revision=revision,
            tokenizer_revision=tokenizer_revision or revision,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,
            trust_remote_code=trust_remote_code,
        )
        self.SamplingParams = SamplingParams
        self.StructuredOutputsParams = StructuredOutputsParams

    @property
    def tokenizer_name(self) -> str:
        return self.model_path

    def generate_until(self, requests):
        request_args = [request.args for request in requests]
        if not request_args:
            return []
        source_prompts = [prompt for prompt, _ in request_args]
        rendered_prompts = [
            render_chat_input(
                prompt,
                self.tokenizer,
                chat_template_kwargs=self.chat_template_kwargs,
                system_message_policy=self.system_message_policy,
            )
            for prompt in source_prompts
        ]
        rendered_requests = [
            (rendered, options)
            for rendered, (_, options) in zip(
                rendered_prompts, request_args, strict=True
            )
        ]
        token_prompts, options = prepare_requests(
            rendered_requests,
            self.tokenizer,
            context_limit=self.max_model_len,
            default_max_tokens=self.max_thinking_tokens,
            add_special_tokens=False,
        )
        outputs1 = self.llm.generate(
            token_prompts,
            [self.SamplingParams(**item) for item in options],
        )
        stage1_outputs = [output.outputs[0].text for output in outputs1]
        if self.stages == 1:
            return stage1_outputs

        reasoning = [reasoning_without_answer(text) for text in stage1_outputs]
        answer_options = {
            "max_gen_toks": 16,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "structured_outputs": self.StructuredOutputsParams(
                regex=self.answer_regex
            ),
        }
        answer_requests = [
            (f"{prompt}{text}\n\nAnswer: ", answer_options)
            for prompt, text in zip(rendered_prompts, reasoning, strict=True)
        ]
        answer_prompts, answer_sampling = prepare_requests(
            answer_requests,
            self.tokenizer,
            context_limit=self.max_model_len,
            default_max_tokens=16,
            add_special_tokens=False,
        )
        outputs2 = self.llm.generate(
            answer_prompts,
            [self.SamplingParams(**item) for item in answer_sampling],
        )
        return [
            finish_answer(
                text,
                output.outputs[0].text,
                self.choices,
                stage1_response=stage1 if self.record_stage1 else None,
            )
            for text, stage1, output in zip(
                reasoning, stage1_outputs, outputs2, strict=True
            )
        ]

    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not supported for staged evaluation")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "loglikelihood_rolling not supported for staged evaluation"
        )


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = typer.Option(..., "--config"),
    output_dir: Path = typer.Option(..., "--output-dir"),
) -> None:
    run_evaluation(Path(config), output_dir=output_dir)


if __name__ == "__main__":
    app()
