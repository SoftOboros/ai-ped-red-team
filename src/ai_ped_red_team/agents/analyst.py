"""Analyst agent wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from ..analyze.metrics import compute_metrics
from ..analyze.report import render_report
from ..analyze.stats import summarize_stats
from ..config import Settings, load_settings


@dataclass
class AnalystAgent:
    """Agent that aggregates metrics, statistics, and reports."""

    settings: Settings = field(default_factory=load_settings)

    def run(
        self,
        results_path: str | Path,
        *,
        summary_path: str | Path | None = None,
        render: bool = False,
    ) -> Dict[str, object]:
        results_file = Path(results_path)
        frame = compute_metrics(results_file)
        summary = summarize_stats(frame)
        target = Path(summary_path) if summary_path else results_file.with_suffix(".summary.json")
        target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if render:
            render_report(target)
        return summary


__all__ = ["AnalystAgent"]
