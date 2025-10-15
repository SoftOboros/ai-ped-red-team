"""Reporting helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from jinja2 import Template
from markdown import markdown

_TEMPLATE_MD = Template(
    """# Analysis Report

Generated at {{ generated_at }}.

## Global metrics
{% if summary.metrics %}
| Metric | Value |
| --- | --- |
{% for metric, value in summary.metrics.items() %}| {{ metric }} | {{ '%.3f'|format(value) }} |
{% endfor %}
{% else %}
_No metrics computed._
{% endif %}

## Metrics by student
{% if summary.by_student %}
{% for student, metrics in summary.by_student.items() %}
### {{ student }}
| Metric | Value |
| --- | --- |
{% for metric, value in metrics.items() %}| {{ metric }} | {{ '%.3f'|format(value) }} |
{% endfor %}
{% endfor %}
{% else %}
_No per-student metrics available._
{% endif %}

## Statistical tests
{% if summary.tests %}
| Metric | t-test p | Mann-Whitney p | Cohen's d | Cliff's δ |
| --- | --- | --- | --- | --- |
{% for metric, stats in summary.tests.items() %}
{% set ttest = '%.3f'|format(stats.get('ttest_p', 0)) %}
{% set mann = '%.3f'|format(stats.get('mannwhitney_p', 0)) %}
{% set cohens = '%.3f'|format(stats.get('cohens_d', 0)) %}
{% set delta = '%.3f'|format(stats.get('cliffs_delta', 0)) %}
| {{ metric }} | {{ ttest }} | {{ mann }} | {{ cohens }} | {{ delta }} |
{% endfor %}
{% else %}
_No statistical tests available._
{% endif %}
"""
)


class ReportRenderError(RuntimeError):
    """Raised when a report cannot be rendered."""


def render_report(summary_path: Path) -> List[Path]:
    """Render Markdown and HTML reports from a summary JSON file."""

    if not summary_path.exists():
        raise ReportRenderError(f"Summary file not found: {summary_path}")

    summary: Dict[str, object] = json.loads(summary_path.read_text())
    context = {"summary": summary, "generated_at": datetime.utcnow().isoformat()}
    markdown_body = _TEMPLATE_MD.render(**context)

    markdown_path = summary_path.with_suffix(".report.md")
    html_path = summary_path.with_suffix(".report.html")
    markdown_path.write_text(markdown_body + "\n", encoding="utf-8")
    html_path.write_text(markdown(markdown_body), encoding="utf-8")

    return [markdown_path, html_path]


__all__ = ["ReportRenderError", "render_report"]
