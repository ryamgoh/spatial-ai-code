"""
SpatialMap cleaner (v6)
=======================
Re-labels SpatialEval JSONL with the shared v6 solver
(`spatial/spatial_solver.py`). Gold matches `spatial/generate_all_v6.py`.

  Type 0: unique compound / two remaining compounds / E. Never A,B,C,D.
  Type 1: proven entities only. No all-four fallback.
  Type 2: definite count.
"""

import json
import sys
from pathlib import Path

import typer

_SPATIAL_DIR = Path(__file__).resolve().parent.parent / "spatial"
if str(_SPATIAL_DIR) not in sys.path:
    sys.path.insert(0, str(_SPATIAL_DIR))
from spatial_solver import parse_options, parse_problem, solve


TYPE0 = """Consider a map with multiple objects:
Photogenic Studio is in the map. Fishing Frenzy is to the Southeast of Photogenic Studio. Gale Gifts is to the Northeast of Photogenic Studio. Gale Gifts is to the Northeast of Fishing Frenzy. Unicorn's Utensils is to the Northeast of Photogenic Studio. Unicorn's Utensils is to the Northeast of Gale Gifts. K University is to the Southwest of Gale Gifts. K University is to the Southwest of Unicorn's Utensils. Peet's Coffee is to the Northwest of Gale Gifts. Peet's Coffee is to the Southwest of Unicorn's Utensils.

Please answer the following multiple-choice question based on the provided information. In which direction is Fishing Frenzy relative to K University? Available options:
A. Northeast
B. Southeast
C. Northwest
D. Southwest."""

TYPE1 = """Consider a map with multiple objects:
Unicorn's Umbrellas is in the map. Eccentric Electronics is to the Northwest of Unicorn's Umbrellas. Mantis's Maps is to the Southeast of Eccentric Electronics. Mantis's Maps is to the Southeast of Unicorn's Umbrellas. Tremor Toys is to the Northwest of Mantis's Maps. Tremor Toys is to the Southwest of Unicorn's Umbrellas. K University is to the Northeast of Eccentric Electronics. K University is to the Northeast of Mantis's Maps. Wild Water Park is to the Southeast of K University. Wild Water Park is to the Northeast of Tremor Toys.

Please answer the following multiple-choice question based on the provided information. Which object is in the Northeast of Unicorn's Umbrellas? Available options:
A. K University
B. Tremor Toys
C. Mantis's Maps
D. Eccentric Electronics."""

TYPE2 = """Consider a map with multiple objects:
Rose Garden Florist is in the map. Eagle's Electronics is to the Southeast of Rose Garden Florist. Albatross's Astronomy Accessories is to the Northeast of Rose Garden Florist. Albatross's Astronomy Accessories is to the Northwest of Eagle's Electronics. K University is to the Southeast of Albatross's Astronomy Accessories. K University is to the Northwest of Eagle's Electronics. Jasmine's Jewellery is to the Southeast of Albatross's Astronomy Accessories. Jasmine's Jewellery is to the Southwest of K University. Fred's Fishing Supplies is to the Northwest of Eagle's Electronics. Fred's Fishing Supplies is to the Northwest of Albatross's Astronomy Accessories.

Please answer the following multiple-choice question based on the provided information. How many objects are in the North of Fred's Fishing Supplies? Available options:
A. 3
B. 2
C. 0
D. 1."""


def clean_jsonl(input_path: str, output_path: str) -> None:
    """Run the v6 solver on every JSONL row and write oracle fields."""
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            text = entry.get("text", "")
            result = solve(text)

            if result.startswith("Error: unknown question type"):
                pass
            elif result.startswith("Error"):
                entry["clean_note"] = result
            elif result == "No valid options found":
                entry["oracle_option"] = ""
                entry["oracle_answer"] = ""
                entry["oracle_full_answer"] = ""
                entry["clean_note"] = result
            else:
                valid_keys = result.split(",")
                _, _, question_part = parse_problem(text)
                options = parse_options(question_part)
                first_key = valid_keys[0]
                first_ans = options.get(first_key, "")
                entry["oracle_option"] = result
                entry["oracle_answer"] = first_ans
                entry["oracle_full_answer"] = " | ".join(
                    f"{k}. {options.get(k, '')}" for k in valid_keys
                )

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if line_no % 500 == 0:
                print(f"  processed {line_no} entries …")

    print(f"Done. Cleaned file written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#   cd eval && uv run python clean_v6.py
#   cd eval && uv run python clean_v6.py batch

app = typer.Typer(help="v6 cleaner: labels JSONL with spatial_solver.SpatialSolver.")


def demo() -> None:
    print(f"Type 0 answer: {solve(TYPE0)}")
    print(f"Type 1 answer: {solve(TYPE1)}")
    print(f"Type 2 answer: {solve(TYPE2)}")


@app.command()
def batch(
    input_file: str = typer.Option(
        "../data/spatialeval_org.jsonl", "--input", "-i",
        help="Raw JSONL to clean.",
    ),
    output_file: str = typer.Option(
        "../data/spatialeval_cleaned_v6.jsonl", "--out", "-o",
        help="Where to write the cleaned JSONL.",
    ),
) -> None:
    print(f"Cleaning {input_file} → {output_file}")
    clean_jsonl(input_file, output_file)


@app.callback(invoke_without_command=True)
def _cli(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        demo()


if __name__ == "__main__":
    app()
