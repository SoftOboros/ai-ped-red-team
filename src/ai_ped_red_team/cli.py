# ruff: noqa: B008
"""Typer CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .analyze.metrics import compute_metrics
from .analyze.report import render_report
from .analyze.stats import summarize_stats
from .config import load_settings
from .generate.variants import generate_variants
from .models.statecheck import ChartEdge, ValidationReport, validate_chart
from .run.runner import RunExecutionConfig, run_variants

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("gen-variants")
def gen_variants_cmd(
    template: Path = typer.Argument(..., help="Path to questionnaire template JSON."),
    hotness: str = typer.Option("cold", help="Temperature profile for variants (cold|hot)."),
    n: int = typer.Option(None, help="Number of variants to request."),
    seed: int = typer.Option(0, help="Seed for deterministic generation."),
    output: Optional[Path] = typer.Option(None, help="Optional path to save variants JSON."),
    ack_hot: bool = typer.Option(
        False,
        "--ack-hot",
        help="Acknowledge safeguards before generating hot variants.",
    ),
):
    """Generate prompt variants and optionally persist them."""

    if hotness == "hot" and not ack_hot:
        raise typer.BadParameter(
            "Pass --ack-hot to confirm safeguards before generating hot variants."
        )
    settings = load_settings()
    variants = generate_variants(template, hotness=hotness, n=n, seed=seed, settings=settings)
    payload = [variant.model_dump() for variant in variants]
    if output:
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(str(output))
    else:
        typer.echo(json.dumps(payload, indent=2))


@app.command()
def run(
    template: Path = typer.Argument(..., help="Path to questionnaire template JSON."),
    ehcp: Path = typer.Argument(..., help="Directory containing EHCP profile text files."),
    model: Optional[str] = typer.Option(None, help="Override tester model."),
    temperature: float = typer.Option(0.0, help="LLM temperature."),
    seed: int = typer.Option(0, help="Seed for completions."),
    n: int = typer.Option(None, help="Number of variants to generate."),
    hotness: str = typer.Option("cold", help="Variant hotness (cold|hot)."),
    counterbalance: bool = typer.Option(True, help="Enable counterbalancing."),
    ack_hot: bool = typer.Option(
        False,
        "--ack-hot",
        help="Acknowledge safeguards before enabling hot prompts.",
    ),
):
    """Run the tester model across EHCP profiles and persist artefacts."""

    if hotness == "hot" and not ack_hot:
        raise typer.BadParameter(
            "Pass --ack-hot to confirm safeguards before running hot variants."
        )
    settings = load_settings()
    config = RunExecutionConfig(
        model=model,
        temperature=temperature,
        seed=seed,
        n_variants=n,
        counterbalance=counterbalance,
        hotness=hotness,
    )
    artefacts = run_variants(template, ehcp, config=config, settings=settings)
    typer.echo(str(artefacts.results_path))
    typer.echo(str(artefacts.csv_path))


@app.command()
def analyze(
    results: Path = typer.Argument(..., help="Path to results JSONL file."),
    output: Optional[Path] = typer.Option(None, help="Where to write the summary JSON."),
    csv: Optional[Path] = typer.Option(
        None,
        "--csv",
        help="Optional path to export metrics as CSV.",
    ),
):
    """Compute metrics and summarise statistics."""

    df = compute_metrics(results)
    summary = summarize_stats(df)
    target = output or results.with_suffix(".summary.json")
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if csv is not None:
        df.to_csv(csv, index=False)
        typer.echo(str(csv))
    typer.echo(str(target))


@app.command()
def report(
    summary: Path = typer.Argument(..., help="Summary JSON path produced by analyze."),
):
    """Render Markdown and HTML reports from a summary file."""

    outputs = render_report(summary)
    for path in outputs:
        typer.echo(str(path))


@app.command("validate-chart")
def validate_chart_cmd(file: Path = typer.Argument(..., help="JSON file with nodes and edges.")):
    """Validate a state chart JSON description."""

    payload = json.loads(file.read_text())
    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []
    edge_models = [ChartEdge.model_validate(edge) for edge in edges]
    report: ValidationReport = validate_chart(nodes, edge_models)
    typer.echo(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    app()
