"""Run orchestration for EHCP experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from ..config import Settings, load_settings
from ..generate.variants import generate_variants
from ..models.gateway import LLMCompletionError, llm_complete
from ..models.schema import PromptVariant, QuestionnaireTemplate, RunConfig, RunResult
from ..templates.loader import load_template


@dataclass
class RunExecutionConfig:
    """Configuration for executing a run."""

    model: Optional[str] = None
    temperature: float = 0.0
    seed: Optional[int] = None
    n_variants: Optional[int] = None
    counterbalance: bool = True
    hotness: str = "cold"


@dataclass
class RunArtifacts:
    """Paths generated during a run."""

    run_dir: Path
    results_path: Path
    csv_path: Path
    variants: List[PromptVariant]
    config: RunConfig


_DEF_RUN_DIR = "reports"


class RunError(RuntimeError):
    """Raised when a run cannot be completed."""



def _read_ehcp_profiles(ehcp_dir: Path) -> List[dict]:
    files = sorted([p for p in ehcp_dir.glob("*.txt") if p.is_file()])
    if not files:
        raise RunError(f"No EHCP text files found in {ehcp_dir}.")
    profiles = []
    for path in files:
        text = path.read_text().strip()
        name = path.stem.replace("_", " ").title()
        summary = text.split(".")[0].strip()
        profiles.append({"name": name, "text": text, "summary": summary, "path": path})
    return profiles



def _render_prompt(template: QuestionnaireTemplate, variant: PromptVariant, profile: dict) -> str:
    base = variant.variant_prompt
    base = base.replace("{{STUDENT_NAME}}", profile["name"])
    base = base.replace("{{SUPPORT_NEED}}", profile["summary"])
    wrapper = template.template_text.replace("{{prompt}}", base)
    prompt_text = (
        f"{wrapper}\n\nEHCP profile for {profile['name']}:\n{profile['text']}"
    )
    return prompt_text.strip()



def _variant_order(counterbalance: bool, index: int) -> List[int]:
    if not counterbalance:
        return [0, 1]
    return [index % 2, (index + 1) % 2]



def _results_to_frame(results: List[RunResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([r.model_dump() for r in results])



def run_variants(
    template_path: str | Path,
    ehcp_dir: str | Path,
    *,
    config: Optional[RunExecutionConfig] = None,
    settings: Optional[Settings] = None,
    variants: Optional[Iterable[PromptVariant]] = None,
) -> RunArtifacts:
    """Execute prompt variants across EHCP profiles."""

    cfg = settings or load_settings()
    template = load_template(template_path)
    run_cfg = config or RunExecutionConfig()
    model_name = run_cfg.model or cfg.tester_model
    count = run_cfg.n_variants or cfg.default_variant_count
    mode = run_cfg.hotness

    variant_list = list(variants) if variants else generate_variants(
        template_path,
        hotness=mode,
        n=count,
        seed=run_cfg.seed or 0,
        settings=cfg,
    )

    profiles = _read_ehcp_profiles(Path(ehcp_dir))
    if len(profiles) < 2:
        raise RunError("Expected at least two EHCP profiles for counterbalancing.")

    results: List[RunResult] = []
    report_root = Path(cfg.reports_dir or _DEF_RUN_DIR)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = report_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    csv_path = run_dir / "results.csv"

    for index, variant in enumerate(variant_list):
        order = _variant_order(run_cfg.counterbalance, index)
        for pos in order:
            profile = profiles[pos % len(profiles)]
            prompt = _render_prompt(template, variant, profile)
            started_at = datetime.utcnow()
            try:
                response = llm_complete(
                    prompt,
                    model=model_name,
                    temperature=run_cfg.temperature,
                    seed=run_cfg.seed,
                    settings=cfg,
                )
                response_text = response.text
                latency_ms = response.latency_ms
                model_info = {
                    "model": response.model,
                    **{k: v for k, v in response.metadata.items() if v is not None},
                }
            except LLMCompletionError as exc:
                response_text = f"ERROR: {exc}"
                latency_ms = 0.0
                model_info = {"model": model_name, "error": str(exc)}
            completed_at = datetime.utcnow()
            result = RunResult(
                variant_id=variant.variant_id,
                student=profile["name"],
                prompt=prompt,
                response=response_text,
                latency_ms=latency_ms,
                model_info=model_info,
                started_at=started_at,
                completed_at=completed_at,
                metadata={
                    "ehcp_path": str(profile["path"]),
                    "variant_position": pos,
                    "variant_hotness": variant.metadata.get("hotness"),
                },
            )
            results.append(result)

    with results_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item.model_dump(), default=str) + os.linesep)

    frame = _results_to_frame(results)
    frame.to_csv(csv_path, index=False)

    run_config = RunConfig(
        model=model_name,
        temperature=run_cfg.temperature,
        seed=run_cfg.seed,
        n_variants=len(variant_list),
        counterbalance=run_cfg.counterbalance,
    )

    return RunArtifacts(
        run_dir=run_dir,
        results_path=results_path,
        csv_path=csv_path,
        variants=variant_list,
        config=run_config,
    )


__all__ = [
    "RunArtifacts",
    "RunError",
    "RunExecutionConfig",
    "run_variants",
]
