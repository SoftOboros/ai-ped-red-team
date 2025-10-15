"""Run orchestration for EHCP experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Optional, Tuple

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
    token_report_path: Path
    token_report_csv: Path
    variants: List[PromptVariant]
    config: RunConfig


_DEF_RUN_DIR = "reports"


class RunError(RuntimeError):
    """Raised when a run cannot be completed."""


def _apply_placeholders(text: str, values: Mapping[str, str]) -> str:
    result = text
    for key, value in values.items():
        if isinstance(value, str):
            result = result.replace(f"{{{{{key}}}}}", value)
    return result


def _read_ehcp_profiles(
    ehcp_dir: Path,
    substitutions: Mapping[str, str] | None = None,
) -> List[dict]:
    files = sorted([p for p in ehcp_dir.glob("*.txt") if p.is_file()])
    if not files:
        raise RunError(f"No EHCP text files found in {ehcp_dir}.")
    profiles = []
    subs = dict(substitutions or {})
    if "SUPPORT_NEED" not in subs and "SUPPORT_NEEDED" in subs:
        subs["SUPPORT_NEED"] = subs["SUPPORT_NEEDED"]
    for path in files:
        text = path.read_text().strip()
        if subs:
            text = _apply_placeholders(text, subs)
        name = subs.get("STUDENT_NAME") or path.stem.replace("_", " ").title()
        summary = (
            subs.get("SUPPORT_NEED") or subs.get("SUPPORT_NEEDED") or text.split(".")[0].strip()
        )
        profiles.append({"name": name, "text": text, "summary": summary, "path": path})
    return profiles


def _render_prompt(
    template: QuestionnaireTemplate,
    variant: PromptVariant,
    profile: dict,
    placeholders: Mapping[str, str] | None = None,
) -> str:
    base = variant.variant_prompt
    mapping = {
        "STUDENT_NAME": profile["name"],
        "SUPPORT_NEED": profile["summary"],
        "SUPPORT_NEEDED": profile["summary"],
    }
    if placeholders:
        mapping.update({k: v for k, v in placeholders.items() if isinstance(v, str)})
    base = _apply_placeholders(base, mapping)
    wrapper = _apply_placeholders(template.template_text, mapping)
    wrapper = wrapper.replace("{{prompt}}", base)
    prompt_text = f"{wrapper}\n\nEHCP profile for {profile['name']}:\n{profile['text']}"
    return prompt_text.strip()


def _build_history_messages(
    history_prompts: Iterable[str] | None,
    prompt_text: str,
    placeholders: Mapping[str, str],
) -> Tuple[List[dict], List[str]]:
    history_texts: List[str] = []
    if not history_prompts:
        return ([{"role": "user", "content": prompt_text}], history_texts)
    messages: List[dict] = []
    for entry in history_prompts:
        role = "user"
        content = entry
        if ":" in entry:
            candidate, remainder = entry.split(":", 1)
            role_candidate = candidate.strip().lower()
            if role_candidate in {"user", "assistant", "system"}:
                role = "assistant" if role_candidate == "assistant" else role_candidate
                content = remainder.strip()
        filled = _apply_placeholders(content, placeholders)
        messages.append({"role": role, "content": filled})
        history_texts.append(f"{role}: {filled}")
    messages.append({"role": "user", "content": prompt_text})
    return messages, history_texts


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
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    substitutions: Optional[Mapping[str, str]] = None,
    history_prompts: Optional[List[str]] = None,
    axes_labels: Optional[Mapping[str, str]] = None,
    run_tag: Optional[str] = None,
) -> RunArtifacts:
    """Execute prompt variants across EHCP profiles."""

    cfg = settings or load_settings()
    template = load_template(template_path)
    run_cfg = config or RunExecutionConfig()
    model_name = run_cfg.model or cfg.tester_model
    count = run_cfg.n_variants or cfg.default_variant_count
    mode = run_cfg.hotness

    variant_list = (
        list(variants)
        if variants
        else generate_variants(
            template_path,
            hotness=mode,
            n=count,
            seed=run_cfg.seed or 0,
            settings=cfg,
        )
    )

    profiles = _read_ehcp_profiles(Path(ehcp_dir), substitutions=substitutions)
    if run_cfg.counterbalance and len(profiles) < 2:
        raise RunError("Expected at least two EHCP profiles for counterbalancing.")

    results: List[RunResult] = []
    token_rows: List[dict] = []
    token_totals = {"invocations": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    orders = [_variant_order(run_cfg.counterbalance, idx) for idx in range(len(variant_list))]
    total_iterations = sum(len(order) for order in orders)
    completed = 0
    if progress_callback:
        progress_callback(0, max(total_iterations, 1), "starting")
    report_root = Path(cfg.reports_dir or _DEF_RUN_DIR)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{run_tag}" if run_tag else ""
    run_dir = report_root / f"{timestamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    csv_path = run_dir / "results.csv"

    for index, variant in enumerate(variant_list):
        order = orders[index]
        for pos in order:
            profile = profiles[pos % len(profiles)]
            placeholder_map = {k: v for k, v in (substitutions or {}).items() if isinstance(v, str)}
            placeholder_map.setdefault("STUDENT_NAME", profile["name"])
            placeholder_map.setdefault("SUPPORT_NEED", profile["summary"])
            placeholder_map.setdefault("SUPPORT_NEEDED", profile["summary"])
            prompt = _render_prompt(template, variant, profile, placeholders=placeholder_map)
            message_sequence, history_texts = _build_history_messages(
                history_prompts,
                prompt,
                placeholder_map,
            )
            started_at = datetime.utcnow()
            try:
                response = llm_complete(
                    prompt,
                    model=model_name,
                    temperature=run_cfg.temperature,
                    seed=run_cfg.seed,
                    settings=cfg,
                    messages=message_sequence,
                )
                response_text = response.text
                latency_ms = response.latency_ms
                model_info = {
                    "model": response.model,
                    **{k: v for k, v in response.metadata.items() if v is not None},
                }
                usage_payload = model_info.get("usage")
                if usage_payload is not None and not isinstance(usage_payload, dict):
                    if hasattr(usage_payload, "model_dump"):
                        model_info["usage"] = usage_payload.model_dump()
                    elif hasattr(usage_payload, "to_dict"):
                        model_info["usage"] = usage_payload.to_dict()
                    elif hasattr(usage_payload, "dict"):
                        model_info["usage"] = usage_payload.dict()
                    elif hasattr(usage_payload, "__dict__"):
                        model_info["usage"] = dict(vars(usage_payload))
                    else:
                        model_info["usage"] = None
            except LLMCompletionError as exc:
                detail = str(exc)
                message = (
                    "LLM request failed after retrying: "
                    f"variant '{variant.variant_id}' / student '{profile['name']}' -- {detail}"
                )
                raise RunError(message) from exc
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
                    "history_labels": list(history_prompts or []),
                    "history_messages": history_texts,
                    "axes": dict(axes_labels or {}),
                },
            )
            results.append(result)

            usage = None
            if isinstance(model_info, dict):
                usage = model_info.get("usage")
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if isinstance(usage, dict):
                prompt_tokens = int(usage.get("prompt_tokens") or usage.get("promptTokens") or 0)
                completion_tokens = int(
                    usage.get("completion_tokens") or usage.get("completionTokens") or 0
                )
                total_tokens = int(
                    usage.get("total_tokens")
                    or usage.get("totalTokens")
                    or (prompt_tokens + completion_tokens)
                )
            if total_tokens:
                token_totals["invocations"] += 1
                token_totals["prompt_tokens"] += prompt_tokens
                token_totals["completion_tokens"] += completion_tokens
                token_totals["total_tokens"] += total_tokens
            token_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "student": profile["name"],
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
            )

            completed += 1
            if progress_callback:
                progress_callback(
                    completed,
                    max(total_iterations, 1),
                    f"{variant.variant_id} -> {profile['name']}",
                )

    with results_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item.model_dump(), default=str) + os.linesep)

    frame = _results_to_frame(results)
    frame.to_csv(csv_path, index=False)

    token_report = {"totals": token_totals, "by_call": token_rows}
    token_report_path = run_dir / "token_usage.json"
    token_report_path.write_text(json.dumps(token_report, indent=2), encoding="utf-8")
    token_report_csv = run_dir / "token_usage.csv"
    token_df = pd.DataFrame(token_rows)
    if not token_df.empty:
        token_df.to_csv(token_report_csv, index=False)
    else:
        token_report_csv.write_text(
            "variant_id,student,prompt_tokens,completion_tokens,total_tokens\n",
            encoding="utf-8",
        )

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
        token_report_path=token_report_path,
        token_report_csv=token_report_csv,
        variants=variant_list,
        config=run_config,
    )


__all__ = [
    "RunArtifacts",
    "RunError",
    "RunExecutionConfig",
    "run_variants",
]
