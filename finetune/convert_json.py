"""
Convert a 'reason'-format spatial training set to Axolotl chat 'messages' format.

Each input item is {system, user, assistant:{reasoning, coordinates, answer}}.
The assistant turn is folded into a single string:
    <thinking>{reasoning}\nCoordinates\n{coords JSON}</thinking>
    Answer: {answer}

Usage:
    cd finetune && uv run python convert_json.py \\
        --input ../data/reason_train.json \\
        --out ../data/reason_train_converted.json
"""

import json
from pathlib import Path

import typer


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


app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_path: str = typer.Option(..., "--input", "-i", help="Source 'reason'-format JSON to convert (required)."),
    output_path: str = typer.Option(..., "--out", "-o", help="Where to write the Axolotl 'messages' JSON (required)."),
) -> None:
    """Convert a 'reason'-format spatial training set into Axolotl chat 'messages' format."""
    input_path = Path(input_path)
    output_path = Path(output_path)

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
    app()
