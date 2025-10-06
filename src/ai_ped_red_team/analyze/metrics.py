"""Metrics computation for run artefacts."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from textstat import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:  # pragma: no cover - optional heavy dependency
    from detoxify import Detoxify
except Exception:  # pragma: no cover - torch/detoxify may be unavailable
    Detoxify = None

try:
    from textstat.backend.counts import _count_syllables
    from textstat.backend.utils import _get_cmudict
except Exception:  # pragma: no cover - textstat internals changed
    _count_syllables = None
    _get_cmudict = None

try:
    textstat.set_language('en_US')
except Exception:  # pragma: no cover - fallback if pyphen missing
    pass

_SENTIMENT_ENV_VAR = 'APRT_SENTIMENT_LEXICON'
_SENTIMENT_FILENAME = 'sentiment_words.json'
_SENTIMENT_HOME_DIR = '.ai_ped_red_team'
_DEFAULT_SENTIMENT = {
    'positive': ['support', 'help', 'encourage', 'improve', 'confidence', 'growth'],
    'negative': ['fail', 'risk', 'concern', 'punish', 'deficit', 'weak'],
}


def _patch_textstat_cmudict() -> None:
    """Disable textstat's cmudict download attempts in offline environments."""

    if _count_syllables is None or _get_cmudict is None:
        return

    def _no_cmudict(lang: str) -> dict[str, list[list[str]]]:
        # Return an empty mapping so downstream syllable logic falls back to Pyphen.
        return {}

    _get_cmudict.get_cmudict = _no_cmudict
    _count_syllables.get_cmudict = _no_cmudict


_patch_textstat_cmudict()


def _candidate_sentiment_paths() -> Tuple[Path, ...]:
    env_path = os.environ.get(_SENTIMENT_ENV_VAR)
    paths = []
    if env_path:
        paths.append(Path(env_path))
    paths.append(Path.cwd() / _SENTIMENT_FILENAME)
    user_root = Path.home() / _SENTIMENT_HOME_DIR
    paths.append(user_root / _SENTIMENT_FILENAME)
    return tuple(paths)


@lru_cache(maxsize=1)
def _ensure_sentiment_file() -> Path:
    paths = _candidate_sentiment_paths()
    for candidate in paths:
        if candidate.exists():
            return candidate
    target = paths[0] if os.environ.get(_SENTIMENT_ENV_VAR) else paths[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_DEFAULT_SENTIMENT, indent=2), encoding='utf-8')
    return target


@lru_cache(maxsize=1)
def _sentiment_word_sets() -> Dict[str, set[str]]:
    path = _ensure_sentiment_file()
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        data = _DEFAULT_SENTIMENT
        path.write_text(json.dumps(_DEFAULT_SENTIMENT, indent=2), encoding='utf-8')
    positive = {str(item).lower() for item in data.get('positive', [])}
    negative = {str(item).lower() for item in data.get('negative', [])}
    return {'positive': positive, 'negative': negative}


@lru_cache(maxsize=1)
def _get_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _get_detoxify() -> Detoxify | None:
    if Detoxify is None:  # pragma: no cover - optional dependency
        return None
    try:
        return Detoxify("original")
    except Exception:  # pragma: no cover - model download/torch failure
        return None


def _detox_predict(text: str) -> Dict[str, float]:
    if not text.strip():
        return {}
    model = _get_detoxify()
    if model is None:
        return {}
    try:
        scores = model.predict(text)
    except Exception:  # pragma: no cover - inference failure
        return {}
    return {str(k): float(v) for k, v in scores.items()}


from ..models.schema import AnalysisRecord
from ..normalize.textnorm import directive_ratio, normalize_text


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
    lexicon = _sentiment_word_sets()
    tokens = [token.lower().strip(".,!") for token in text.split()]
    if not tokens:
        return 0.0
    pos = sum(1 for token in tokens if token in lexicon['positive'])
    neg = sum(1 for token in tokens if token in lexicon['negative'])
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
        vader_score = _get_vader().polarity_scores(response or "").get('compound', 0.0)
        heuristic_score = _sentiment_heuristic(response)

        metadata = entry.get("metadata") or {}
        history_messages = metadata.get("history_messages") or []
        history_text = "\n".join(str(msg) for msg in history_messages if msg)

        detox_response = _detox_predict(response)
        detox_history = _detox_predict(history_text) if history_text else {}
        detox_keys = (
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack",
        )

        record_kwargs = {
            "variant_id": entry.get("variant_id", "unknown"),
            "student": entry.get("student", "unknown"),
            "word_count": word_count,
            "char_count": char_count,
            "directive_ratio": directive_ratio(response),
            "readability": readability,
            "sentiment": vader_score,
        }

        extra = {
            "started_at": entry.get("started_at"),
            "completed_at": entry.get("completed_at"),
            "model": entry.get("model_info", {}).get("model"),
            "lexicon_sentiment": heuristic_score,
            "detoxify_response": detox_response,
            "detoxify_history": detox_history,
        }

        for key in detox_keys:
            resp_value = detox_response.get(key)
            hist_value = detox_history.get(key)
            record_kwargs[key] = resp_value
            record_kwargs[f"history_{key}"] = hist_value
            record_kwargs[f"{key}_delta"] = (
                resp_value - hist_value if resp_value is not None and hist_value is not None else None
            )

        record_kwargs["extra"] = extra

        record = AnalysisRecord(**record_kwargs)
        records.append(record)

    return pd.DataFrame([r.model_dump() for r in records])


__all__ = ["compute_metrics"]
