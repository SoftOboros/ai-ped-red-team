# ruff: noqa: B008
"""Typer CLI entrypoint."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .analyze.metrics import compute_metrics
from .analyze.report import render_report
from .analyze.stats import summarize_stats
from .axes.config import AxesConfigError, load_axes_config
from .config import load_settings
from .generate.variants import generate_variants
from .models.statecheck import ChartEdge, ValidationReport, validate_chart
from .run.runner import RunExecutionConfig, run_variants

app = typer.Typer(no_args_is_help=True, add_completion=False)

console = Console()
progress_columns = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    TimeElapsedColumn(),
)


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
    with console.status("Generating variants...", spinner="dots"):
        variants = generate_variants(
            template, hotness=hotness, n=n, seed=seed, settings=settings
        )
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
    axes_config: Optional[Path] = typer.Option(
        None,
        help="Optional TOML describing axes (gender/support/history) to exhaust combinations.",
    ),
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
    axes_runs = [({}, {})]
    axes_labels: list[dict[str, str]] = [{}]
    if axes_config is not None:
        try:
            axes = load_axes_config(axes_config)
        except AxesConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc
        axes_runs = []
        axes_labels = []
        for axis_mapping, values in axes.iter_combinations():
            axes_runs.append((axis_mapping, values))
            axes_labels.append({axis: option.label for axis, option in axis_mapping.items()})

    artefact_paths = []
    for (axis_mapping, value_map), label_map in zip(axes_runs, axes_labels):
        history_raw = value_map.get("HISTORY_PROMPTS")
        history_prompts = None
        if isinstance(history_raw, list):
            history_prompts = [str(item) for item in history_raw]

        substitutions = {
            k: v
            for k, v in value_map.items()
            if isinstance(v, str) and not k.startswith("HISTORY_")
        }
        if "SUPPORT_NEED" not in substitutions and "SUPPORT_NEEDED" in substitutions:
            substitutions["SUPPORT_NEED"] = substitutions["SUPPORT_NEEDED"]

        tag_parts = [f"{axis}-{option.label}" for axis, option in axis_mapping.items()]
        run_tag = "-".join(tag_parts)
        run_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", run_tag).strip("_") or None

        with Progress(*progress_columns, console=console) as progress:
            title = run_tag or "baseline"
            task_id = progress.add_task(f"Preparing run ({title})", total=1)

            def _update_progress(completed: int, total: int, description: str) -> None:
                if progress.tasks[task_id].total != total:
                    progress.update(task_id, total=total)
                progress.update(task_id, completed=completed, description=description)

            artefacts = run_variants(
                template,
                ehcp,
                config=config,
                settings=settings,
                progress_callback=_update_progress,
                substitutions=substitutions,
                history_prompts=history_prompts,
                axes_labels=label_map,
                run_tag=run_tag,
            )
            final_total = progress.tasks[task_id].total or progress.tasks[task_id].completed or 1
            progress.update(task_id, completed=final_total, description="Run complete")
        artefact_paths.append(artefacts)

    for artefacts in artefact_paths:
        typer.echo(str(artefacts.results_path))
        typer.echo(str(artefacts.csv_path))
        typer.echo(str(artefacts.token_report_path))
        typer.echo(str(artefacts.token_report_csv))


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

    with console.status("Computing metrics...", spinner="dots"):
        df = compute_metrics(results)
        summary = summarize_stats(df)
    target = output or results.with_suffix(".summary.json")
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if csv is not None:
        df.to_csv(csv, index=False)
        typer.echo(str(csv))
    typer.echo(str(target))


@app.command()
def wizard() -> None:
    """Interactive guide from template to report."""

    console.rule("[bold cyan]AI Ped Red Team Wizard")
    base_settings = load_settings()

    vendor_input = typer.prompt("Model vendor", default="openai").strip()
    model_input = typer.prompt("Model name", default="gpt-5-nano").strip()
    if "/" in model_input:
        full_model = model_input
    elif vendor_input:
        full_model = f"{vendor_input}/{model_input}"
    else:
        full_model = model_input

    settings = base_settings.model_copy(
        update={
            "generator_model": full_model,
            "tester_model": full_model,
            "analyst_model": full_model,
        }
    )

    default_template = Path(
        "src/ai_ped_red_team/templates/examples/questionnaire/q_ehcp_gender.json"
    )
    template_default_str = str(default_template) if default_template.exists() else ""

    def _prompt_template_path(initial: Path) -> Path:
        current = initial
        while True:
            if current.is_file():
                return current
            if current.is_dir():
                candidates = sorted(current.glob("*.json"))
                if candidates:
                    table = Table(title="Templates in directory")
                    table.add_column("#", justify="right")
                    table.add_column("Path", overflow="fold")
                    for idx, candidate in enumerate(candidates, start=1):
                        table.add_row(str(idx), str(candidate))
                    console.print(table)
                    selection = typer.prompt("Choose template number", default="1").strip()
                    try:
                        index = int(selection) - 1
                        if 0 <= index < len(candidates):
                            return candidates[index]
                    except ValueError:
                        pass
                    console.print("[red]Invalid selection. Try again.")
                else:
                    console.print(
                        f"[red]Directory {current} contains no JSON templates."
                    )
            else:
                console.print(f"[red]Template not found: {current}")
            template_input = typer.prompt(
                "Template path", default=str(current)
            ).strip()
            if not template_input:
                template_input = str(current)
            current = Path(template_input).expanduser()

    template_input = typer.prompt(
        "Template path", default=template_default_str
    ).strip()
    if not template_input:
        template_input = template_default_str
    template_path = _prompt_template_path(Path(template_input).expanduser())

    default_ehcp_dir = Path("src/ai_ped_red_team/templates/examples/ehcp_pair")
    ehcp_default_str = str(default_ehcp_dir) if default_ehcp_dir.exists() else ""
    ehcp_input = typer.prompt("EHCP directory", default=ehcp_default_str)
    ehcp_dir = Path(ehcp_input).expanduser()
    while not ehcp_dir.exists() or not ehcp_dir.is_dir():
        console.print(f"[red]EHCP directory not found: {ehcp_dir}")
        ehcp_input = typer.prompt("EHCP directory", default=str(ehcp_dir))
        ehcp_dir = Path(ehcp_input).expanduser()

    hotness = typer.prompt("Variant hotness (cold/hot)", default="cold").strip().lower()
    while hotness not in {"cold", "hot"}:
        console.print("[red]Please choose 'cold' or 'hot'.")
        hotness = typer.prompt("Variant hotness (cold/hot)", default="cold").strip().lower()

    if hotness == "hot" and not typer.confirm(
        "Hot variants can be provocative. Continue?",
        default=False,
    ):
        console.print("[yellow]Aborting wizard at user request.")
        raise typer.Exit(code=1)

    try:
        n_variants = int(
            typer.prompt(
                "How many variants should we generate?",
                default=str(settings.default_variant_count),
            )
        )
    except ValueError:
        console.print("[yellow]Invalid number; falling back to default.")
        n_variants = settings.default_variant_count

    try:
        temperature = float(typer.prompt("Tester temperature", default="0.0"))
    except ValueError:
        temperature = 0.0

    try:
        seed = int(typer.prompt("Random seed", default="0"))
    except ValueError:
        seed = 0

    model_override_input = typer.prompt(
        "Tester model (press enter for default)",
        default=full_model,
    ).strip()
    model_override = model_override_input or full_model

    with console.status("Generating variants...", spinner="dots"):
        variants = generate_variants(
            template_path,
            hotness=hotness,
            n=n_variants,
            seed=seed,
            settings=settings,
        )

    table = Table(title="Prompt Variants")
    table.add_column("ID", style="cyan")
    table.add_column("Prompt")
    for variant in variants:
        preview = variant.variant_prompt
        preview = preview if len(preview) <= 120 else f"{preview[:117]}..."
        table.add_row(variant.variant_id, preview)
    console.print(table)

    config = RunExecutionConfig(
        model=model_override,
        temperature=temperature,
        seed=seed,
        n_variants=n_variants,
        counterbalance=True,
        hotness=hotness,
    )

    with Progress(*progress_columns, console=console) as progress:
        task_id = progress.add_task("Running variants", total=len(variants) * 2 or 1)

        def _update_progress(completed: int, total: int, description: str) -> None:
            if progress.tasks[task_id].total != total:
                progress.update(task_id, total=total)
            progress.update(task_id, completed=completed, description=description)

        artefacts = run_variants(
            template_path,
            ehcp_dir,
            config=config,
            settings=settings,
            variants=variants,
            progress_callback=_update_progress,
        )
        final_total = progress.tasks[task_id].total or progress.tasks[task_id].completed or 1
        progress.update(task_id, completed=final_total, description="Runs complete")

    console.print(f"[green]Results written to: {artefacts.results_path}")
    console.print(f"[green]Tabular log: {artefacts.csv_path}")
    console.print(f"[green]Token usage (JSON): {artefacts.token_report_path}")
    console.print(f"[green]Token usage (CSV): {artefacts.token_report_csv}")

    token_summary = json.loads(artefacts.token_report_path.read_text())
    totals = token_summary.get("totals", {})
    if totals:
        prompt_tokens = totals.get("prompt_tokens", 0)
        completion_tokens = totals.get("completion_tokens", 0)
        total_tokens = totals.get("total_tokens", 0)
        invocation_count = totals.get("invocations", 0)
        console.print("[cyan]Tokens summary:")
        console.print(
            f"[cyan]  prompt: {prompt_tokens} | completion: {completion_tokens}"
        )
        console.print(
            f"[cyan]  total: {total_tokens} (calls: {invocation_count})"
        )

    with console.status("Analyzing results...", spinner="dots"):
        metrics_frame = compute_metrics(artefacts.results_path)
        summary = summarize_stats(metrics_frame)
        summary_path = artefacts.results_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        metrics_csv = artefacts.run_dir / "metrics.csv"
        metrics_frame.to_csv(metrics_csv, index=False)

    console.print(f"[green]Metrics CSV: {metrics_csv}")
    console.print(f"[green]Summary JSON: {summary_path}")

    if summary.get("metrics"):
        summary_table = Table(title="Average Metrics")
        summary_table.add_column("Metric", style="magenta")
        summary_table.add_column("Value", justify="right")
        for metric, value in summary["metrics"].items():
            summary_table.add_row(metric, f"{value:.3f}")
        console.print(summary_table)

    with console.status("Rendering report...", spinner="dots"):
        report_paths = render_report(summary_path)

    console.print("[cyan]Reports generated:")
    for path in report_paths:
        console.print(f"  - {path}")

    console.rule("[bold green]Wizard complete")


@app.command()
def report(
    summary: Path = typer.Argument(..., help="Summary JSON path produced by analyze."),
):
    """Render Markdown and HTML reports from a summary file."""

    with console.status("Rendering report...", spinner="dots"):
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
