"""
Convert reason_train.json to Axolotl-compatible messages format.

Usage:
    cd finetune && uv run python convert_json.py
"""

import json
from pathlib import Path


def convert_item(item: dict) -> dict:
    reasoning = item["assistant"]["reasoning"]
    coords = item["assistant"]["coordinates"]
    answer = item["assistant"]["answer"]

    thinking = reasoning + "\nCoordinates\n" + json.dumps(coords, ensure_ascii=False);

    structured = f"<think>{thinking}</think>Answer: {answer}";

    return {
        "messages": [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": structured},
        ]
    }


def main():
    input_path = Path(__file__).parent.parent / "reason_train.json"
    output_path = Path(__file__).parent.parent / "reason_train_converted.json"

    print(f"Loading: {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    print(f"Converting {len(data)} samples...")
    converted = [convert_item(item) for item in data]

    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")
    print(f"Sample output:\n{'-' * 50}")
    print(f"Messages: {converted[0]['messages']}")


if __name__ == "__main__":
    main()
