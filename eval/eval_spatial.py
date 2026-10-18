"""Text-only staged evaluation for spatially post-trained checkpoints.

Standard checkpoints reuse ``VLLMStagedChatPass``. SenseNova needs a
trust-remote-code loader, kept here so the frozen general and reasoning runner
hashes remain unchanged.
"""

from __future__ import annotations

from pathlib import Path

import typer
from eval_comparators import VLLMStagedChatPass  # noqa: F401
from eval_new import run_evaluation
from generation_contract import answer_regex
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("vllm_staged_remote_chat_pass")
class VLLMStagedRemoteChatPass(VLLMStagedChatPass):
    """The shared staged decoder with a remote-code model loader."""

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
        chat_template_kwargs: dict[str, object] | None = None,
        system_message_policy: str = "native",
        **_: object,
    ) -> None:
        LM.__init__(self)
        if stages not in (1, 2):
            raise ValueError(f"stages must be 1 or 2, got {stages}")
        if not revision:
            raise ValueError("spatial evaluation requires a pinned revision")

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
            trust_remote_code=True,
            use_fast=False,
        )
        self.llm = LLM(
            model=pretrained,
            revision=revision,
            tokenizer_revision=tokenizer_revision or revision,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,
            trust_remote_code=True,
        )
        self.SamplingParams = SamplingParams
        self.StructuredOutputsParams = StructuredOutputsParams


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = typer.Option(..., "--config"),
    output_dir: Path = typer.Option(..., "--output-dir"),
) -> None:
    run_evaluation(Path(config), output_dir=output_dir)


if __name__ == "__main__":
    app()
