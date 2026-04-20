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

from cotorra.extractor import Extractor
from cotorra.trainer import Trainer
from cotorra.tuner import Tuner

app = typer.Typer(
    name="cotorra", help="Configurable training for generative event models"
)
console = Console()


@app.command()
def train(
    main_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--main-config", "-m", help="Main configuration file (overrides default)"
        ),
    ] = None,
    model_config: Annotated[
        Optional[pathlib.Path],
        typer.Option("--model-config", help="Model configuration file"),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
        ),
    ] = None,
    output_home: Annotated[
        Optional[str],
        typer.Option("--output-home", "-o", help="Output directory for trained models"),
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
        trainer = Trainer(
            main_cfg=main_config,
            mdl_cfg=model_config,
            processed_data_home=processed_data_home,
            output_home=output_home,
        )
        trainer.train(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Training completed in {t1 - t0:.2f}s.")
        out_path = trainer.output_home / f"mdl-{trainer.cfg.run_name}"
        print(f"  Model: [cyan]{out_path}[/cyan]")


@app.command()
def tune(
    main_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--main-config", "-m", help="Main configuration file (overrides default)"
        ),
    ] = None,
    model_config: Annotated[
        Optional[pathlib.Path],
        typer.Option("--model-config", help="Model configuration file"),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
        ),
    ] = None,
    output_home: Annotated[
        Optional[str],
        typer.Option("--output-home", "-o", help="Output directory for trained models"),
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
        tuner = Tuner(
            main_cfg=main_config,
            mdl_cfg=model_config,
            processed_data_home=processed_data_home,
            output_home=output_home,
        )
        tuner.train(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Tuning completed in {t1 - t0:.2f}s.")
        out_path = tuner.output_home / f"mdl-{tuner.cfg.run_name}"
        print(f"  Model: [cyan]{out_path}[/cyan]")


@app.command()
def extract(
    main_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--main-config", "-m", help="Main configuration file (overrides default)"
        ),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
        ),
    ] = None,
    output_home: Annotated[
        Optional[str],
        typer.Option("--output-home", "-o", help="Output directory for trained models"),
    ] = None,
):
    """
    Extract representations from a trained model.
    """
    with console.status("[bold green]Extracting representations..."):
        t0 = time.perf_counter()
        extractor = Extractor(
            main_cfg=main_config,
            processed_data_home=processed_data_home,
            output_home=output_home,
        )
        extractor.extract()
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Extraction completed in {t1 - t0:.2f}s.")
        for split in extractor.loader.splits:
            output = extractor.processed_data_home / f"features-{split}.parquet"
            print(f" Output: {output}")


def main():
    app()


if __name__ == "__main__":
    main()
