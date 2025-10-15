# ruff: noqa: B008
"""Typer CLI entrypoint."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .analyze.metrics import compute_metrics
from .analyze.report import render_report
from .analyze.stats import summarize_stats
from .axes.config import AxesConfigError, AxisOption, load_axes_config
from .config import load_settings
from .generate.variants import generate_variants
from .models.statecheck import ChartEdge, ValidationReport, validate_chart
from .run.runner import RunArtifacts, RunError, RunExecutionConfig, run_variants

app = typer.Typer(no_args_is_help=True, add_completion=False)

_DEBUG_ENABLED = False
_DEBUG_HINT = "Re-run with --debug or set LITELLM_LOG=debug to inspect LiteLLM request logs."


def _print_run_error(exc: RunError) -> None:
    console.print(f"[red]{exc}")
    if not _DEBUG_ENABLED:
        console.print(f"[yellow]{_DEBUG_HINT}")


console = Console()
progress_columns = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    TimeElapsedColumn(),
)

_VENDOR_ALIAS = {
    "google": "gemini",
    "gemini": "gemini",
    "openai": "openai",
    "anthropic": "anthropic",
    "mistral": "mistral",
    "cohere": "cohere",
    "openrouter": "openrouter",
}


def _vendor_status(settings) -> List[Tuple[str, str, bool]]:
    mapping = {
        "openai": ("OPENAI_API_KEY", settings.openai_api_key),
        "google": ("GOOGLE_API_KEY", getattr(settings, "google_api_key", None)),
        "anthropic": ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
        "openrouter": ("OPENROUTER_API_KEY", settings.openrouter_api_key),
        "mistral": ("MISTRAL_API_KEY", settings.mistral_api_key),
        "cohere": ("COHERE_API_KEY", settings.cohere_api_key),
    }
    return [(vendor, env_var, bool(value)) for vendor, (env_var, value) in mapping.items()]


def _configured_vendors(settings) -> List[str]:
    return [vendor for vendor, _, present in _vendor_status(settings) if present]


@app.callback(invoke_without_command=True)
def main_callback(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable LiteLLM debug logging for outbound requests.",
    ),
) -> None:
    """Global CLI options."""

    global _DEBUG_ENABLED
    if debug and not _DEBUG_ENABLED:
        try:  # pragma: no cover - optional dependency path
            import litellm

            litellm._turn_on_debug()
            console.print("[yellow]LiteLLM debug logging enabled.")
            _DEBUG_ENABLED = True
        except Exception as exc:
            console.print(f"[red]Failed to enable LiteLLM debug logging: {exc}")


def _list_openai_models(settings, limit: int) -> List[str]:
    if not settings.openai_api_key:
        configured = _configured_vendors(settings)
        suffix = (
            f"Configured vendors: {', '.join(configured)}"
            if configured
            else "No vendors configured."
        )
        raise typer.BadParameter(f"OPENAI_API_KEY is not set. {suffix}")
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise typer.BadParameter(f"OpenAI SDK unavailable: {exc}") from exc
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        models = client.models.list()
    except Exception as exc:
        raise typer.BadParameter(f"Failed to list OpenAI models: {exc}") from exc
    return [item.id for item in models.data][:limit]


def _list_google_models(settings, limit: int) -> List[str]:
    api_key = getattr(settings, "google_api_key", None)
    if not api_key:
        configured = _configured_vendors(settings)
        suffix = (
            f"Configured vendors: {', '.join(configured)}"
            if configured
            else "No vendors configured."
        )
        raise typer.BadParameter(f"GOOGLE_API_KEY is not set. {suffix}")
    params = {"key": api_key, "pageSize": min(limit, 100)}
    try:
        response = httpx.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params=params,
            timeout=15,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise typer.BadParameter(f"Failed to list Gemini models: {exc}") from exc
    data = response.json()
    return [
        item.get("name", "").split("/")[-1] for item in data.get("models", []) if item.get("name")
    ][:limit]


def _list_anthropic_models(settings, limit: int) -> List[str]:
    api_key = settings.anthropic_api_key
    if not api_key:
        configured = _configured_vendors(settings)
        suffix = (
            f"Configured vendors: {', '.join(configured)}"
            if configured
            else "No vendors configured."
        )
        raise typer.BadParameter(f"ANTHROPIC_API_KEY is not set. {suffix}")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    try:
        response = httpx.get(
            "https://api.anthropic.com/v1/models",
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise typer.BadParameter(f"Failed to list Anthropic models: {exc}") from exc
    payload = response.json()
    items = payload.get("data") or []
    results = []
    for item in items:
        model_id = item.get("id") or item.get("model")
        if model_id:
            results.append(model_id)
        if len(results) >= limit:
            break
    return results


def _list_mistral_models(settings, limit: int) -> List[str]:
    api_key = settings.mistral_api_key
    if not api_key:
        configured = _configured_vendors(settings)
        suffix = (
            f"Configured vendors: {', '.join(configured)}"
            if configured
            else "No vendors configured."
        )
        raise typer.BadParameter(f"MISTRAL_API_KEY is not set. {suffix}")
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = httpx.get(
            "https://api.mistral.ai/v1/models",
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise typer.BadParameter(f"Failed to list Mistral models: {exc}") from exc
    payload = response.json()
    items = payload.get("data") or payload.get("models") or []
    results: List[str] = []
    for item in items:
        model_id = item.get("id") or item.get("name")
        if model_id:
            results.append(model_id)
        if len(results) >= limit:
            break
    return results


def _list_cohere_models(settings, limit: int) -> List[str]:
    api_key = settings.cohere_api_key
    if not api_key:
        configured = _configured_vendors(settings)
        suffix = (
            f"Configured vendors: {', '.join(configured)}"
            if configured
            else "No vendors configured."
        )
        raise typer.BadParameter(f"COHERE_API_KEY is not set. {suffix}")
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = httpx.get(
            "https://api.cohere.com/v1/models",
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise typer.BadParameter(f"Failed to list Cohere models: {exc}") from exc
    payload = response.json()
    items = payload.get("models") or payload.get("data") or []
    results: List[str] = []
    for item in items:
        if isinstance(item, str):
            results.append(item)
        else:
            model_id = item.get("id") or item.get("name")
            if model_id:
                results.append(model_id)
        if len(results) >= limit:
            break
    return results


def _list_vendor_models(vendor: str, settings, limit: int) -> List[str]:
    canonical = _VENDOR_ALIAS.get(vendor.lower(), vendor.lower())
    if canonical == "openai":
        return _list_openai_models(settings, limit)
    if canonical == "gemini":
        return _list_google_models(settings, limit)
    if canonical == "anthropic":
        return _list_anthropic_models(settings, limit)
    if canonical == "mistral":
        return _list_mistral_models(settings, limit)
    if canonical == "cohere":
        return _list_cohere_models(settings, limit)
    raise typer.BadParameter(f"Model listing not implemented for vendor '{vendor}'.")


@app.command()
def vendors() -> None:
    """List supported vendors and credential status."""

    settings = load_settings()
    table = Table(title="Vendor credentials", show_lines=False)
    table.add_column("Vendor", style="cyan")
    table.add_column("Env Var", style="magenta")
    table.add_column("Configured", style="green")
    for vendor, env_var, present in _vendor_status(settings):
        table.add_row(vendor, env_var, "yes" if present else "no")
    console.print(table)


@app.command()
def models(
    vendor: str = typer.Argument("*", help="Vendor name, or '*' to list supported vendors."),
    limit: int = typer.Option(50, help="Maximum number of models to display."),
) -> None:
    """List models from a provider."""

    settings = load_settings()
    if vendor == "*":
        console.print("Use `aprt models <vendor>` with one of:")
        for vendor_name, _, present in _vendor_status(settings):
            status = "configured" if present else "missing-key"
            console.print(f" - {vendor_name} ({status})")
        return

    model_ids = _list_vendor_models(vendor, settings, limit)
    if not model_ids:
        console.print(f"No models returned for vendor '{vendor}'.")
        return

    table = Table(title=f"Models for {vendor}")
    table.add_column("#", justify="right")
    table.add_column("Model ID", overflow="fold")
    for idx, model_id in enumerate(model_ids, start=1):
        table.add_row(str(idx), model_id)
    console.print(table)


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
    for (axis_mapping, value_map), label_map in zip(axes_runs, axes_labels, strict=False):
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

            def _update_progress(
                completed: int, total: int, description: str, task_id=task_id
            ) -> None:
                if progress.tasks[task_id].total != total:
                    progress.update(task_id, total=total)
                progress.update(task_id, completed=completed, description=description)

            try:
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
            except RunError as exc:
                progress.update(
                    task_id,
                    completed=progress.tasks[task_id].completed,
                    description="error",
                )
                _print_run_error(exc)
                raise typer.Exit(code=1) from exc

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

    vendor_input = typer.prompt("Model vendor", default="openai").strip() or "openai"
    vendor_lower = vendor_input.lower()
    canonical_vendor = _VENDOR_ALIAS.get(vendor_lower, vendor_lower)

    status_entries = {
        vendor: (env_var, present) for vendor, env_var, present in _vendor_status(base_settings)
    }
    env_var, present = status_entries.get(
        vendor_lower, status_entries.get(canonical_vendor, (None, True))
    )
    if not present and env_var:
        console.print(
            f"[yellow]Warning: {vendor_lower} credentials not configured (set {env_var})."
        )
        configured = _configured_vendors(base_settings)
        if configured:
            console.print(f"[yellow]Configured vendors: {', '.join(configured)}")
        else:
            console.print("[yellow]No providers currently have API keys configured.")
        if not typer.confirm(f"Continue anyway with vendor '{vendor_lower}'?", default=False):
            console.print("[yellow]Aborting wizard at user request.")
            raise typer.Exit(code=1)

    default_model = "gpt-5-nano" if canonical_vendor == "openai" else "gemini-pro"
    vendor_models: List[str] = []
    try:
        vendor_models = _list_vendor_models(canonical_vendor, base_settings, limit=10)
        if vendor_models:
            default_model = vendor_models[0]
    except typer.BadParameter as exc:
        console.print(f"[yellow]{exc}. Using manual entry.")
        vendor_models = []

    if not default_model:
        default_model = "gpt-5-nano" if canonical_vendor == "openai" else "gemini-pro"

    while True:
        model_input = typer.prompt(
            "Model name (enter '*' to list available models)",
            default=default_model,
        ).strip()
        if model_input == "*":
            try:
                vendor_models = _list_vendor_models(canonical_vendor, base_settings, limit=50)
            except typer.BadParameter as exc:
                console.print(f"[red]{exc}")
                continue
            if not vendor_models:
                console.print("[yellow]No models available from this vendor.")
                continue
            table = Table(title=f"Models for {vendor_input}")
            table.add_column("#", justify="right")
            table.add_column("Model ID", overflow="fold")
            for idx, model_id in enumerate(vendor_models, start=1):
                table.add_row(str(idx), model_id)
            console.print(table)
            continue
        if not model_input:
            model_input = default_model
        simple_model = model_input.split("/", 1)[-1] if "/" in model_input else model_input
        if canonical_vendor == "cohere":
            lower_name = simple_model.lower()
            if lower_name.startswith("embed") or lower_name.startswith("rerank"):
                console.print(
                    "[yellow]Selected Cohere model appears to be embedding-only. "
                    "Choose a generative model such as 'command-r-plus'."
                )
                continue
        break

    if "/" in model_input:
        full_model = model_input
    elif canonical_vendor:
        full_model = f"{canonical_vendor}/{model_input}"
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
                    console.print(f"[red]Directory {current} contains no JSON templates.")
            else:
                console.print(f"[red]Template not found: {current}")
            template_input = typer.prompt("Template path", default=str(current)).strip()
            if not template_input:
                template_input = str(current)
            current = Path(template_input).expanduser()

    template_input = typer.prompt("Template path", default=template_default_str).strip()
    if not template_input:
        template_input = template_default_str
    template_path = _prompt_template_path(Path(template_input).expanduser())

    default_ehcp_dir = Path("examples/EHCP-templates")
    if not default_ehcp_dir.exists():
        default_ehcp_dir = Path("src/ai_ped_red_team/templates/examples/ehcp_pair")
    ehcp_default_str = str(default_ehcp_dir) if default_ehcp_dir.exists() else ""
    ehcp_input = typer.prompt("EHCP directory", default=ehcp_default_str)
    ehcp_dir = Path(ehcp_input).expanduser()
    while not ehcp_dir.exists() or not ehcp_dir.is_dir():
        console.print(f"[red]EHCP directory not found: {ehcp_dir}")
        ehcp_input = typer.prompt("EHCP directory", default=str(ehcp_dir))
        ehcp_dir = Path(ehcp_input).expanduser()

    axes_runs: list[tuple[dict[str, AxisOption], dict[str, object]]] = [({}, {})]
    axes_labels: list[dict[str, str]] = [{}]
    axes_config_path: Path | None = None
    default_axes_path = Path("examples/ehcp_variables.toml")
    axes_default_str = str(default_axes_path) if default_axes_path.exists() else ""
    axes_input = typer.prompt(
        "Axes config (enter to skip)",
        default=axes_default_str,
    ).strip()
    if axes_input:
        candidate = Path(axes_input).expanduser()
        while not candidate.exists():
            console.print(f"[red]Axes config not found: {candidate}")
            axes_input = typer.prompt("Axes config (enter to skip)", default="").strip()
            if not axes_input:
                candidate = None
                break
            candidate = Path(axes_input).expanduser()
        if candidate:
            try:
                axes = load_axes_config(candidate)
            except AxesConfigError as exc:
                console.print(f"[yellow]Failed to load axes config: {exc}")
                if not typer.confirm("Continue without axes config?", default=True):
                    console.print("[yellow]Aborting wizard at user request.")
                    raise typer.Exit(code=1) from None
            else:
                axes_config_path = candidate
                axes_runs = []
                axes_labels = []
                for axis_mapping, values in axes.iter_combinations():
                    axes_runs.append((dict(axis_mapping), dict(values)))
                    axes_labels.append(
                        {axis: option.label for axis, option in axis_mapping.items()}
                    )

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
    run_records: list[tuple[RunArtifacts, dict[str, str], str | None]] = []

    for (axis_mapping, value_map), label_map in zip(axes_runs, axes_labels, strict=False):
        substitutions = {
            k: v
            for k, v in value_map.items()
            if isinstance(v, str) and not k.startswith("HISTORY_")
        }
        if "SUPPORT_NEED" not in substitutions and "SUPPORT_NEEDED" in substitutions:
            substitutions["SUPPORT_NEED"] = substitutions["SUPPORT_NEEDED"]
        history_prompts = None
        history_raw = value_map.get("HISTORY_PROMPTS")
        if isinstance(history_raw, list):
            history_prompts = [str(item) for item in history_raw]

        tag_parts = [f"{axis}-{option.label}" for axis, option in axis_mapping.items()]
        run_tag = "-".join(tag_parts)
        run_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", run_tag).strip("_") or None

        with Progress(*progress_columns, console=console) as progress:
            title = run_tag or "baseline"
            if axes_config_path and label_map:
                pretty_axes = ", ".join(f"{axis}={label}" for axis, label in label_map.items())
                title = f"{title} ({pretty_axes})"
            task_id = progress.add_task(f"Running variants ({title})", total=len(variants) * 2 or 1)

            def _update_progress(
                completed: int, total: int, description: str, task_id=task_id
            ) -> None:
                if progress.tasks[task_id].total != total:
                    progress.update(task_id, total=total)
                progress.update(task_id, completed=completed, description=description)

            try:
                artefacts = run_variants(
                    template_path,
                    ehcp_dir,
                    config=config,
                    settings=settings,
                    variants=variants,
                    progress_callback=_update_progress,
                    substitutions=substitutions or None,
                    history_prompts=history_prompts,
                    axes_labels=label_map,
                    run_tag=run_tag,
                )
            except RunError as exc:
                progress.update(
                    task_id,
                    completed=progress.tasks[task_id].completed,
                    description="error",
                )
                _print_run_error(exc)
                raise typer.Exit(code=1) from exc

            final_total = progress.tasks[task_id].total or progress.tasks[task_id].completed or 1
            progress.update(task_id, completed=final_total, description="Runs complete")

        run_records.append((artefacts, label_map, run_tag))

    for artefacts, label_map, _run_tag in run_records:
        if axes_config_path and label_map:
            axis_descriptor = ", ".join(f"{axis}={label}" for axis, label in label_map.items())
            console.rule(f"[bold green]Run artefacts ({axis_descriptor})")
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
            console.print(f"[cyan]  prompt: {prompt_tokens} | completion: {completion_tokens}")
            console.print(f"[cyan]  total: {total_tokens} (calls: {invocation_count})")

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
            title = "Average Metrics"
            if label_map:
                title = f"Average Metrics ({', '.join(f'{k}={v}' for k, v in label_map.items())})"
            summary_table = Table(title=title)
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
