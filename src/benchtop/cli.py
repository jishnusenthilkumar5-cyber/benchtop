"""benchtop command line entrypoint."""

from __future__ import annotations

import typer

from benchtop.commands import collect, dash, train
from benchtop.commands import eval as eval_cmd

app = typer.Typer(
    help="Reproducible evaluation for robot manipulation policies.", no_args_is_help=True
)
app.add_typer(collect.app, name="collect")
app.add_typer(train.app, name="train")
app.add_typer(eval_cmd.app, name="eval")
app.add_typer(dash.app, name="dash")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
