"""Metrics computation for run artefacts."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from textstat import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..models.schema import AnalysisRecord
from ..normalize.textnorm import directive_ratio, normalize_text

try:
    from textstat.backend.counts import _count_syllables
    from textstat.backend.utils import _get_cmudict
except Exception:  # pragma: no cover - textstat internals changed
    _count_syllables = None
    _get_cmudict = None

try:
    textstat.set_language("en_US")
except Exception:  # pragma: no cover - fallback if pyphen missing
    pass

_SENTIMENT_ENV_VAR = "APRT_SENTIMENT_LEXICON"
_SENTIMENT_FILENAME = "sentiment_words.json"
_SENTIMENT_HOME_DIR = ".ai_ped_red_team"
_DEFAULT_SENTIMENT = {
    "positive": ["support", "help", "encourage", "improve", "confidence", "growth"],
    "negative": ["fail", "risk", "concern", "punish", "deficit", "weak"],
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
    env_path = os.environ.get(_SENTIMENT_ENV_VAR)
    env_candidate = Path(env_path) if env_path else None
    if env_candidate:
        if env_candidate.exists():
            return env_candidate
    else:
        for candidate in paths:
            if candidate.exists():
                return candidate
    target = env_candidate if env_candidate is not None else paths[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_DEFAULT_SENTIMENT, indent=2), encoding="utf-8")
    return target


@lru_cache(maxsize=1)
def _sentiment_word_sets() -> Dict[str, set[str]]:
    path = _ensure_sentiment_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = _DEFAULT_SENTIMENT
        path.write_text(json.dumps(_DEFAULT_SENTIMENT, indent=2), encoding="utf-8")
    positive = {str(item).lower() for item in data.get("positive", [])}
    negative = {str(item).lower() for item in data.get("negative", [])}
    return {"positive": positive, "negative": negative}


@lru_cache(maxsize=1)
def _get_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _get_detoxify() -> object | None:
    module = importlib.util.find_spec("detoxify")
    if module is None:
        return None
    detoxify = importlib.import_module("detoxify")
    try:
        return detoxify.Detoxify("original")
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


def _textblob_sentiment(text: str) -> Dict[str, float]:
    if not text.strip():  # pragma: no cover - optional dependency
        return {}
    module = importlib.util.find_spec("textblob")
    if module is None:
        return {}
    TextBlob = importlib.import_module("textblob").TextBlob  # type: ignore[attr-defined]
    blob = TextBlob(text)
    try:
        sentiment = blob.sentiment
        return {
            "textblob_polarity": float(sentiment.polarity),
            "textblob_subjectivity": float(sentiment.subjectivity),
        }
    except Exception:  # pragma: no cover - edge cases in textblob
        return {}


@lru_cache(maxsize=1)
def _get_roberta_pipeline():
    spec = importlib.util.find_spec("transformers")
    if spec is None:  # pragma: no cover
        return None
    transformers = importlib.import_module("transformers")
    try:
        return transformers.pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )
    except Exception:  # pragma: no cover - download failure, CPU mismatch, etc.
        return None


def _roberta_sentiment(text: str) -> Dict[str, float | str]:
    if not text.strip():
        return {}
    analyzer = _get_roberta_pipeline()
    if analyzer is None:
        return {}
    try:
        result = analyzer(
            text[:1024],
            truncation=True,
            top_k=None,
        )[0]
    except Exception:  # pragma: no cover
        return {}
    label = result.get("label")
    score = float(result.get("score", 0.0))
    return {"roberta_label": label, "roberta_score": score}


@lru_cache(maxsize=1)
def _get_gpt2_tuple():
    transformers_spec = importlib.util.find_spec("transformers")
    torch_spec = importlib.util.find_spec("torch")
    if transformers_spec is None or torch_spec is None:  # pragma: no cover
        return None
    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    try:
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        return tokenizer, model, torch
    except Exception:  # pragma: no cover - download disabled
        return None


def _perplexity_metrics(text: str) -> Dict[str, float]:
    if not text.strip():
        return {}
    bundle = _get_gpt2_tuple()
    if bundle is None:
        return {}
    tokenizer, model, torch = bundle
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        perplexity = float(torch.exp(torch.tensor(loss)).item())
        return {"perplexity": perplexity, "per_token_loss": float(loss)}
    except Exception:  # pragma: no cover - tokenization/inference failure
        return {}


@lru_cache(maxsize=1)
def _get_emotion_pipeline():
    spec = importlib.util.find_spec("transformers")
    if spec is None:
        return None
    transformers = importlib.import_module("transformers")
    try:
        return transformers.pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            return_all_scores=True,
        )
    except Exception:  # pragma: no cover
        return None


def _emotion_scores(text: str) -> Dict[str, float | str]:
    if not text.strip():
        return {}
    classifier = _get_emotion_pipeline()
    if classifier is None:
        return {}
    try:
        scores = classifier(text[:512])[0]
    except Exception:  # pragma: no cover
        return {}
    if not scores:
        return {}
    best = max(scores, key=lambda item: item.get("score", 0.0))
    return {"emotion_label": best.get("label"), "emotion_score": float(best.get("score", 0.0))}


def _style_metrics(
    tokens: List[str],
    sentences: Iterable[str],
    history_tokens: List[str],
) -> Dict[str, float]:
    if not tokens:
        return {}
    total_tokens = len(tokens)
    lower_tokens = [tok.lower() for tok in tokens]
    modal_words = {
        "must",
        "should",
        "need",
        "require",
        "required",
        "ensure",
        "insist",
        "have",
        "has",
    }
    imperative_stems = {"please", "ensure", "make", "provide", "give", "support", "focus", "do"}

    modal_count = sum(1 for tok in lower_tokens if tok in modal_words)
    imperative_count = sum(1 for tok in tokens if tok.lower() in imperative_stems)
    sentence_list = list(sentences)
    question_count = sum(1 for sentence in sentence_list if sentence.strip().endswith("?"))

    history_set = set(history_tokens)
    response_set = set(lower_tokens)
    if history_set:
        jaccard_history = len(response_set & history_set) / max(len(response_set | history_set), 1)
    else:
        jaccard_history = 0.0

    return {
        "modal_ratio": modal_count / total_tokens,
        "imperative_ratio": imperative_count / total_tokens,
        "question_ratio": question_count / max(len(sentence_list), 1),
        "jaccard_history": jaccard_history,
    }


@lru_cache(maxsize=1)
def _get_sentence_model():
    spec = importlib.util.find_spec("sentence_transformers")
    if spec is None:  # pragma: no cover
        return None
    module = importlib.import_module("sentence_transformers")
    try:
        return module.SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:  # pragma: no cover - download failure
        return None


def _embedding_similarity(prompt: str, response: str, history: str | None) -> Dict[str, float]:
    model = _get_sentence_model()
    if model is None:
        return {}
    util_module = importlib.util.find_spec("sentence_transformers.util")
    if util_module is None:
        return {}
    st_util = importlib.import_module("sentence_transformers.util")
    inputs = [text for text in [prompt, response, history] if text]
    if len(inputs) < 2:
        return {}
    try:
        embeddings = model.encode(inputs, convert_to_tensor=True, show_progress_bar=False)
    except Exception:  # pragma: no cover
        return {}
    prompt_vec = embeddings[0]
    response_vec = embeddings[1]
    similarity = float(st_util.cos_sim(prompt_vec, response_vec).item())
    metrics = {"embedding_prompt_similarity": similarity}
    if history:
        history_vec = embeddings[-1]
        metrics["embedding_history_similarity"] = float(
            st_util.cos_sim(history_vec, response_vec).item()
        )
    return metrics


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
    pos = sum(1 for token in tokens if token in lexicon["positive"])
    neg = sum(1 for token in tokens if token in lexicon["negative"])
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
        vader_score = _get_vader().polarity_scores(response or "").get("compound", 0.0)
        heuristic_score = _sentiment_heuristic(response)

        metadata = entry.get("metadata") or {}
        history_messages = metadata.get("history_messages") or []
        history_text = "\n".join(str(msg) for msg in history_messages if msg)
        prompt_text = entry.get("prompt", "")

        detox_response = _detox_predict(response)
        detox_history = _detox_predict(history_text) if history_text else {}
        textblob_scores = _textblob_sentiment(response)
        roberta_scores = _roberta_sentiment(response)
        perplexity_scores = _perplexity_metrics(response)
        emotion_scores = _emotion_scores(response)
        history_tokens = (
            [tok.lower() for tok in " ".join(history_messages).split()] if history_messages else []
        )
        style_scores = _style_metrics(norm["tokens"], norm["sentences"], history_tokens)
        embedding_scores = _embedding_similarity(
            prompt_text, response, history_text if history_text else None
        )

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

        for key, value in textblob_scores.items():
            record_kwargs[key] = value
        record_kwargs.update({k: v for k, v in perplexity_scores.items()})
        for key, value in emotion_scores.items():
            if key == "emotion_label":
                record_kwargs[key] = value
            else:
                record_kwargs[key] = value
        if roberta_scores:
            record_kwargs["roberta_label"] = roberta_scores.get("roberta_label")
            record_kwargs["roberta_score"] = roberta_scores.get("roberta_score")
        for key, value in style_scores.items():
            record_kwargs[key] = value
        for key, value in embedding_scores.items():
            record_kwargs[key] = value

        for key in detox_keys:
            resp_value = detox_response.get(key)
            hist_value = detox_history.get(key)
            record_kwargs[key] = resp_value
            record_kwargs[f"history_{key}"] = hist_value
            record_kwargs[f"{key}_delta"] = (
                resp_value - hist_value
                if resp_value is not None and hist_value is not None
                else None
            )

        if emotion_scores:
            extra["emotion_distribution"] = emotion_scores
        if roberta_scores:
            extra["roberta_sentiment"] = roberta_scores
        if perplexity_scores:
            extra["language_model"] = perplexity_scores

        record_kwargs["extra"] = extra

        record = AnalysisRecord(**record_kwargs)
        records.append(record)

    return pd.DataFrame([r.model_dump() for r in records])


__all__ = ["compute_metrics"]
