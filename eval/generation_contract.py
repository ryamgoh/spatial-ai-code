"""Token budgets and answer normalization shared by both inference stages."""

import re


def prepare_requests(
    requests, tokenizer, *, context_limit, default_max_tokens, add_special_tokens
):
    token_prompts, sampling = [], []
    for prompt, options in requests:
        max_tokens = options.get("max_gen_toks", default_max_tokens)
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("generation budget must be a positive integer")
        token_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        if len(token_ids) + max_tokens > context_limit:
            raise ValueError("prompt plus generation budget exceeds model context")
        token_prompts.append({"prompt_token_ids": token_ids})
        sampling.append(
            {
                "max_tokens": max_tokens,
                "temperature": options.get("temperature", 0.6),
                "repetition_penalty": options.get("repetition_penalty", 1.1),
                "stop": options.get("until") or None,
                **{
                    key: options[key]
                    for key in ("top_p", "top_k", "seed", "structured_outputs")
                    if key in options
                },
            }
        )
    return token_prompts, sampling


def reasoning_without_answer(completion):
    """Remove earlier standalone answer footers before the constrained re-ask."""
    return re.sub(r"(?m)^\s*Answer:[^\n]*", "", completion).rstrip()


def finish_answer(reasoning, answer, choices):
    letters = [part.strip() for part in answer.split(",")]
    if not letters or any(letter not in choices for letter in letters):
        raise ValueError("answer stage returned an invalid option letter")
    canonical = ", ".join(sorted(set(letters)))
    return f"{reasoning_without_answer(reasoning)}\n\nAnswer: {canonical}"
