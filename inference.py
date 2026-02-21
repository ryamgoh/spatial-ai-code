"""
Run inference on a fine-tuned model.

Usage:
    uv run inference --model ./outputs/qwen3-7b-lora-finetuned --prompt "Your question here"
    uv run inference --model ./outputs/qwen3-7b-lora-finetuned --interactive
    uv run inference --model Qwen/Qwen2.5-7B-Instruct --prompt "What is 2+2?"

vLLM automatically handles LoRA adapter loading when you pass a path to a fine-tuned model.
"""

import argparse
import sys
from pathlib import Path

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a fine-tuned model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace ID (works with LoRA adapters)",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive chat mode"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (lower = more deterministic)",
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="qwen3",
        choices=["qwen3", "chatml", "llama3"],
        help="Chat template format",
    )
    return parser.parse_args()


def format_prompt(user_input: str, system_prompt: str, template: str) -> str:
    """Format prompt with chat template."""
    if template == "qwen3":
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    elif template == "chatml":
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    elif template == "llama3":
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        return f"System: {system_prompt}\nUser: {user_input}\nAssistant: "


def run_single_prompt(llm: LLM, prompt: str, sampling_params: SamplingParams) -> str:
    """Run a single prompt and return the response."""
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text


def run_interactive(
    llm: LLM, sampling_params: SamplingParams, system_prompt: str, template: str
) -> None:
    """Run interactive chat mode."""
    print("\n" + "=" * 50)
    print("Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        prompt = format_prompt(user_input, system_prompt, template)
        response = run_single_prompt(llm, prompt, sampling_params)
        print(f"\nAssistant: {response}\n")


def main():
    args = parse_args()

    if not args.prompt and not args.interactive:
        print("Error: Please provide --prompt or use --interactive")
        print("\nExamples:")
        print(
            "  uv run inference --model Qwen/Qwen2.5-7B-Instruct --prompt 'What is 2+2?'"
        )
        print("  uv run inference --model ./outputs/my-model --interactive")
        sys.exit(1)

    model_path = Path(args.model)
    if model_path.exists():
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from HuggingFace: {args.model}")

    llm = LLM(model=args.model)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.interactive:
        run_interactive(
            llm=llm,
            sampling_params=sampling_params,
            system_prompt=args.system,
            template=args.chat_template,
        )
    elif args.prompt:
        prompt = format_prompt(args.prompt, args.system, args.chat_template)
        response = run_single_prompt(llm, prompt, sampling_params)
        print(response)


if __name__ == "__main__":
    main()
