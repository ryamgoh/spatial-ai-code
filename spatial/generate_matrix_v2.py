"""CLI for generating a SpatialMap V2 ablation matrix."""

from pathlib import Path

import typer
from spatial_matrix_v2 import generate_matrix

app = typer.Typer(add_completion=False)


@app.command()
def main(
    matrix: Path = typer.Argument(..., help="YAML experiment matrix."),  # noqa: B008
    out: Path = typer.Option(..., help="Output JSONL base path."),  # noqa: B008
    replace: bool = typer.Option(False, help="Replace this matrix's existing outputs."),
) -> None:
    try:
        train_path, test_path, manifest_path = generate_matrix(
            matrix,
            out,
            replace=replace,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"train: {train_path}")
    typer.echo(f"test: {test_path}")
    typer.echo(f"manifest: {manifest_path}")


if __name__ == "__main__":
    app()
