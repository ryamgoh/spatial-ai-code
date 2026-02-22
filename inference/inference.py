"""
Run inference on a fine-tuned model with optional RAG and LoRA adapter support.

Usage:
    cd inference && uv run python inference.py --base-model Qwen/Qwen2.5-7B-Instruct --prompt "What is 2+2?"

    cd inference && uv run python inference.py --base-model Qwen/Qwen3-7B --adapter ./outputs/qwen3-7b-lora-finetuned --prompt "test"

    cd inference && uv run python inference.py --base-model Qwen/Qwen3-7B --adapter ./outputs/adapter --interactive

    cd inference && uv run python inference.py --base-model Qwen/Qwen2.5-7B-Instruct --prompt "What is X?" --rag --corpus ./datasets/
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag import RAGRetriever


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a model")
    parser.add_argument(
        "--base-model", required=True, help="Base model path or HuggingFace ID"
    )
    parser.add_argument(
        "--adapter", default=None, help="Path to LoRA adapter (optional)"
    )

    parser.add_argument("--prompt", default=None, help="Single prompt to run")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive chat mode"
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--system", default="You are a helpful assistant.")

    parser.add_argument(
        "--rag", action="store_true", help="Enable RAG for context retrieval"
    )
    parser.add_argument(
        "--corpus", default=None, help="Path to corpus file or directory for RAG"
    )
    parser.add_argument(
        "--context-k", type=int, default=3, help="Number of context chunks to retrieve"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=800, help="Chunk size for RAG"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=100, help="Chunk overlap for RAG"
    )
    return parser.parse_args()


def load_model(base_model_name: str, adapter_path: str | None = None):
    """Load base model and optionally apply LoRA adapter."""

    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def format_messages(
    prompt: str, system_prompt: str, context: str | None = None
) -> list[dict]:
    """Format messages for chat template."""
    user_content = prompt
    if context:
        user_content = f"Context:\n{context}\n\nQuestion: {prompt}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def run_single_prompt(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    context: str | None = None,
) -> str:
    """Run a single prompt and return the response."""

    messages = format_messages(prompt, system_prompt, context)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    elif "assistant" in response.lower():
        parts = response.lower().split("assistant")
        response = response[len("".join(parts[:-1])) + len("assistant") :].strip()

    return response


def run_interactive(
    model,
    tokenizer,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    retriever: RAGRetriever | None = None,
    context_k: int = 3,
) -> None:
    """Run interactive chat mode with optional RAG."""
    print("\n" + "=" * 50)
    print("Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    if retriever:
        print("RAG: Enabled")
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

        context = None
        if retriever:
            context = retriever.get_context(query=user_input, k=context_k)

        response = run_single_prompt(
            model,
            tokenizer,
            user_input,
            system_prompt,
            max_tokens,
            temperature,
            context,
        )
        print(f"\nAssistant: {response}\n")


def main():
    args = parse_args()

    if not args.prompt and not args.interactive:
        print("Error: Please provide --prompt or use --interactive")
        print("\nExamples:")
        print(
            "  uv run python inference.py --base-model Qwen/Qwen2.5-7B-Instruct --prompt 'What is 2+2?'"
        )
        print(
            "  uv run python inference.py --base-model Qwen/Qwen3-7B --adapter ./outputs/adapter --interactive"
        )
        print(
            "  uv run python inference.py --base-model Qwen/Qwen2.5-7B-Instruct --prompt 'X?' --rag --corpus ./datasets/"
        )
        sys.exit(1)

    retriever = None
    if args.rag and args.corpus:
        print(f"Setting up RAG with corpus: {args.corpus}")
        retriever = RAGRetriever(
            corpus_paths=[args.corpus],
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        retriever.build_index()
    elif args.rag and not args.corpus:
        print("Warning: --rag enabled but no --corpus specified. RAG disabled.")

    model, tokenizer = load_model(args.base_model, args.adapter)

    if args.interactive:
        run_interactive(
            model=model,
            tokenizer=tokenizer,
            system_prompt=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            retriever=retriever,
            context_k=args.context_k,
        )
    elif args.prompt:
        context = None
        if retriever:
            context = retriever.get_context(query=args.prompt, k=args.context_k)

        response = run_single_prompt(
            model,
            tokenizer,
            args.prompt,
            args.system,
            args.max_tokens,
            args.temperature,
            context,
        )
        print(response)


if __name__ == "__main__":
    main()
