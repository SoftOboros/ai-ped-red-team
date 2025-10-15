"""Statistical summaries for analysis metrics."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

_NUMERIC_COLUMNS = [
    "word_count",
    "char_count",
    "directive_ratio",
    "readability",
    "sentiment",
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "history_toxicity",
    "history_severe_toxicity",
    "history_obscene",
    "history_threat",
    "history_insult",
    "history_identity_attack",
    "toxicity_delta",
    "severe_toxicity_delta",
    "obscene_delta",
    "threat_delta",
    "insult_delta",
    "identity_attack_delta",
    "textblob_polarity",
    "textblob_subjectivity",
    "perplexity",
    "per_token_loss",
    "emotion_score",
    "modal_ratio",
    "imperative_ratio",
    "question_ratio",
    "jaccard_history",
    "embedding_prompt_similarity",
    "embedding_history_similarity",
    "roberta_score",
]


class StatsComputationError(RuntimeError):
    """Raised when stats cannot be computed."""


def _benjamini_hochberg(p_values: List[Tuple[str, str, float]]) -> Dict[Tuple[str, str], float]:
    """Apply Benjamini-Hochberg correction to a list of p-values."""

    if not p_values:
        return {}
    sorted_vals = sorted(
        ((metric, test, p) for metric, test, p in p_values),
        key=lambda item: item[2],
    )
    n = len(sorted_vals)
    adjusted: Dict[Tuple[str, str], float] = {}
    prev = 1.0
    for rank, (metric, test, p_value) in enumerate(sorted_vals, start=1):
        value = min(prev, p_value * n / rank)
        prev = value
        adjusted[(metric, test)] = value
    return adjusted


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    mean_diff = a.mean() - b.mean()
    numerator = ((len(a) - 1) * a.var(ddof=1)) + ((len(b) - 1) * b.var(ddof=1))
    denominator = len(a) + len(b) - 2
    if denominator <= 0 or numerator <= 0:
        return 0.0
    pooled_var = numerator / denominator
    return float(mean_diff / np.sqrt(pooled_var))


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    greater = sum(x > y for x in a for y in b)
    lesser = sum(x < y for x in a for y in b)
    total = len(a) * len(b)
    if total == 0:
        return 0.0
    return float((greater - lesser) / total)


def _global_summary(df: pd.DataFrame) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for column in _NUMERIC_COLUMNS:
        if column in df and df[column].notna().any():
            summary[column] = float(df[column].mean())
    return summary


def _by_student(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, float]] = {}
    if "student" not in df:
        return grouped
    for student, frame in df.groupby("student"):
        grouped[student] = {}
        for column in _NUMERIC_COLUMNS:
            if column in frame and frame[column].notna().any():
                grouped[student][column] = float(frame[column].mean())
    return grouped


def _two_sample_tests(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if "student" not in df or df["student"].nunique() != 2:
        return {}
    tests_summary: Dict[str, Dict[str, float]] = {}
    students = sorted(df["student"].unique())
    a_frame, b_frame = (df[df["student"] == students[0]], df[df["student"] == students[1]])
    raw_pvalues: List[Tuple[str, str, float]] = []
    for column in _NUMERIC_COLUMNS:
        if column not in df:
            continue
        series_a = a_frame[column].dropna()
        series_b = b_frame[column].dropna()
        if len(series_a) < 2 or len(series_b) < 2:
            continue
        try:
            a_values = series_a.to_numpy(dtype=float)
            b_values = series_b.to_numpy(dtype=float)
            t_stat, t_pvalue = stats.ttest_ind(a_values, b_values, equal_var=False)
            u_stat, u_pvalue = stats.mannwhitneyu(a_values, b_values, alternative="two-sided")
            effect_d = _cohens_d(a_values, b_values)
            effect_delta = _cliffs_delta(a_values, b_values)
            tests_summary[column] = {
                "ttest_stat": float(t_stat),
                "ttest_p": float(t_pvalue),
                "mannwhitney_stat": float(u_stat),
                "mannwhitney_p": float(u_pvalue),
                "cohens_d": effect_d,
                "cliffs_delta": effect_delta,
            }
            raw_pvalues.extend(
                [
                    (column, "ttest_p", t_pvalue),
                    (column, "mannwhitney_p", u_pvalue),
                ]
            )
        except ValueError:
            continue
    adjusted = _benjamini_hochberg(raw_pvalues)
    for column, stats_dict in tests_summary.items():
        for key in ["ttest_p", "mannwhitney_p"]:
            adj = adjusted.get((column, key))
            if adj is not None:
                stats_dict[f"{key}_adj"] = adj
    return tests_summary


def summarize_stats(df: pd.DataFrame) -> Dict[str, object]:
    """Summarise metrics with descriptive and inferential statistics."""

    if df.empty:
        return {"metrics": {}, "by_student": {}, "tests": {}, "note": "No data available."}

    return {
        "metrics": _global_summary(df),
        "by_student": _by_student(df),
        "tests": _two_sample_tests(df),
    }


__all__ = ["StatsComputationError", "summarize_stats"]
