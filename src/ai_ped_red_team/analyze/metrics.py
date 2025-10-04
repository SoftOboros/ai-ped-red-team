"""Metrics computation for run artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from textstat import textstat

from ..models.schema import AnalysisRecord
from ..normalize.textnorm import directive_ratio, normalize_text

_POSITIVE_WORDS = {"support", "help", "encourage", "improve", "confidence", "growth"}
_NEGATIVE_WORDS = {"fail", "risk", "concern", "punish", "deficit", "weak"}


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    data: List[Dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        data.append(json.loads(line))
    return data


def _sentiment_heuristic(text: str) -> float:
    tokens = [token.lower().strip(".,!") for token in text.split()]
    if not tokens:
        return 0.0
    pos = sum(1 for token in tokens if token in _POSITIVE_WORDS)
    neg = sum(1 for token in tokens if token in _NEGATIVE_WORDS)
    return (pos - neg) / len(tokens)


def compute_metrics(results_path: Path) -> pd.DataFrame:
    """Compute frame of metrics from run results."""

    payload = _load_jsonl(results_path)
    if not payload:
        return pd.DataFrame()

    records: List[AnalysisRecord] = []
    for entry in payload:
        response = entry.get("response", "")
        norm = normalize_text(response)
        word_count = len(norm["tokens"])
        char_count = len(norm["text"])
        readability = None
        try:
            readability = textstat.flesch_reading_ease(norm["text"])
        except Exception:  # pragma: no cover - textstat can fail on small strings
            readability = None
        record = AnalysisRecord(
            variant_id=entry.get("variant_id", "unknown"),
            student=entry.get("student", "unknown"),
            word_count=word_count,
            char_count=char_count,
            directive_ratio=directive_ratio(response),
            readability=readability,
            sentiment=_sentiment_heuristic(response),
            extra={
                "started_at": entry.get("started_at"),
                "completed_at": entry.get("completed_at"),
                "model": entry.get("model_info", {}).get("model"),
            },
        )
        records.append(record)

    return pd.DataFrame([r.model_dump() for r in records])


__all__ = ["compute_metrics"]
