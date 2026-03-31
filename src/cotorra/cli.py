#!/usr/bin/env python3

"""
CLI for cotorra - configurable training for generative event models
"""

import pathlib
import time
from typing import Annotated, Optional

import typer
from rich import print
from rich.console import Console

from cotorra.trainer import Trainer
from cotorra.tuner import Tuner

app = typer.Typer(
    name="cotorra", help="Configurable training for generative event models"
)
console = Console()


@app.command()
def train(
    output: Annotated[
        Optional[pathlib.Path],
        typer.Option("--output", "-o", help="Output directory for collated data"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Verbose logging for collate", is_flag=True
        ),
    ] = False,
):
    """
    Train a model on tokenized data. For tokenization, consult the cocoa package.
    """
    with console.status("[bold green]Training model..."):
        t0 = time.perf_counter()
        trainer = Trainer() if output is None else Trainer(output_dir=output)
        trainer.train(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Training completed in {t1 - t0:.2f}s.")


@app.command()
def tune(
    output: Annotated[
        Optional[pathlib.Path],
        typer.Option("--output", "-o", help="Output directory for collated data"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Verbose logging for collate", is_flag=True
        ),
    ] = False,
):
    """
    Run hyperparameter tuning while training a model.
    """
    with console.status("[bold green]Tuning model..."):
        t0 = time.perf_counter()
        tuner = Tuner() if output is None else Tuner(output_dir=output)
        tuner.train(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Tuning completed in {t1 - t0:.2f}s.")


def main():
    app()


if __name__ == "__main__":
    main()
