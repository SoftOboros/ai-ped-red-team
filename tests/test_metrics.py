import json

import pandas as pd
import pytest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ai_ped_red_team.analyze import metrics
from ai_ped_red_team.analyze.metrics import compute_metrics


def test_compute_metrics_handles_empty(tmp_path):
    data = tmp_path / "results.jsonl"
    data.write_text("")
    df = compute_metrics(data)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_compute_metrics_offline_nltk(monkeypatch, tmp_path):
    results = tmp_path / "results.jsonl"
    payload = {
        "variant_id": "v1",
        "student": "Test Student",
        "response": "Support the student with positive feedback.",
    }
    results.write_text(json.dumps(payload) + "\n")

    lex_path = tmp_path / "lexicon.json"
    monkeypatch.setenv("APRT_SENTIMENT_LEXICON", str(lex_path))
    metrics._sentiment_word_sets.cache_clear()
    metrics._ensure_sentiment_file.cache_clear()

    calls = []

    def _fail_download(*args, **kwargs):
        calls.append(args)
        raise AssertionError("nltk.download should not be called")

    import nltk

    monkeypatch.setattr(nltk, "download", _fail_download)

    detox_scores = {
        "toxicity": 0.12,
        "severe_toxicity": 0.01,
        "obscene": 0.0,
        "threat": 0.0,
        "insult": 0.02,
        "identity_attack": 0.0,
    }
    monkeypatch.setattr(
        metrics,
        "_detox_predict",
        lambda text, s=detox_scores: s.copy(),
    )

    monkeypatch.setattr(
        metrics,
        "_textblob_sentiment",
        lambda text: {"textblob_polarity": 0.1, "textblob_subjectivity": 0.2},
    )
    monkeypatch.setattr(
        metrics,
        "_roberta_sentiment",
        lambda text: {"roberta_label": "positive", "roberta_score": 0.9},
    )
    monkeypatch.setattr(
        metrics,
        "_perplexity_metrics",
        lambda text: {"perplexity": 10.0, "per_token_loss": 2.3},
    )
    monkeypatch.setattr(
        metrics,
        "_emotion_scores",
        lambda text: {"emotion_label": "joy", "emotion_score": 0.75},
    )
    monkeypatch.setattr(
        metrics,
        "_style_metrics",
        lambda tokens, sentences, history_tokens: {
            "modal_ratio": 0.1,
            "imperative_ratio": 0.05,
            "question_ratio": 0.0,
            "jaccard_history": 0.0,
        },
    )
    monkeypatch.setattr(
        metrics,
        "_embedding_similarity",
        lambda prompt, resp, hist: {"embedding_prompt_similarity": 0.8},
    )

    df = compute_metrics(results)
    assert not df.empty
    assert not calls
    assert lex_path.exists()


def test_compute_metrics_uses_vader(monkeypatch, tmp_path):
    text = "Support the student but highlight risk"  # includes positive and negative cues
    payload = {
        "variant_id": "v1",
        "student": "Test Student",
        "response": text,
    }
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(payload) + "\n")

    lex_path = tmp_path / "custom_lexicon.json"
    monkeypatch.setenv("APRT_SENTIMENT_LEXICON", str(lex_path))
    metrics._sentiment_word_sets.cache_clear()
    metrics._ensure_sentiment_file.cache_clear()

    detox_scores = {
        "toxicity": 0.33,
        "severe_toxicity": 0.02,
        "obscene": 0.0,
        "threat": 0.0,
        "insult": 0.05,
        "identity_attack": 0.0,
    }
    monkeypatch.setattr(
        metrics,
        "_detox_predict",
        lambda text, s=detox_scores: s.copy(),
    )
    monkeypatch.setattr(
        metrics,
        "_textblob_sentiment",
        lambda text: {"textblob_polarity": 0.2, "textblob_subjectivity": 0.4},
    )
    monkeypatch.setattr(
        metrics,
        "_roberta_sentiment",
        lambda text: {"roberta_label": "neutral", "roberta_score": 0.6},
    )
    monkeypatch.setattr(
        metrics,
        "_perplexity_metrics",
        lambda text: {"perplexity": 20.0, "per_token_loss": 3.0},
    )
    monkeypatch.setattr(
        metrics,
        "_emotion_scores",
        lambda text: {"emotion_label": "anger", "emotion_score": 0.5},
    )
    monkeypatch.setattr(
        metrics,
        "_style_metrics",
        lambda tokens, sentences, history_tokens: {
            "modal_ratio": 0.2,
            "imperative_ratio": 0.1,
            "question_ratio": 0.25,
            "jaccard_history": 0.3,
        },
    )
    monkeypatch.setattr(
        metrics,
        "_embedding_similarity",
        lambda prompt, resp, hist: {"embedding_prompt_similarity": 0.75},
    )

    df = compute_metrics(results)
    assert not df.empty

    analyzer = SentimentIntensityAnalyzer()
    expected_vader = analyzer.polarity_scores(text)["compound"]
    assert df.iloc[0]["sentiment"] == pytest.approx(expected_vader)

    lexicon = json.loads(lex_path.read_text())
    tokens = [token.lower().strip(".,!") for token in text.split()]
    pos = sum(1 for token in tokens if token in lexicon["positive"])
    neg = sum(1 for token in tokens if token in lexicon["negative"])
    expected_heuristic = (pos - neg) / len(tokens)
    assert df.iloc[0]["extra"]["lexicon_sentiment"] == pytest.approx(expected_heuristic)


def test_compute_metrics_detox_history(monkeypatch, tmp_path):
    entry = {
        "variant_id": "v_history",
        "student": "Jordan",
        "response": "The plan supports positive routines.",
        "metadata": {
            "history_messages": [
                "user: Please enforce compliance.",
                "assistant: I will emphasise obedience.",
            ]
        },
    }
    results = tmp_path / "history.jsonl"
    results.write_text(json.dumps(entry) + "\n")

    def fake_detox(text: str):
        if "obedience" in text:
            return {
                "toxicity": 0.4,
                "severe_toxicity": 0.05,
                "obscene": 0.0,
                "threat": 0.01,
                "insult": 0.1,
                "identity_attack": 0.02,
            }
        return {
            "toxicity": 0.1,
            "severe_toxicity": 0.01,
            "obscene": 0.0,
            "threat": 0.0,
            "insult": 0.02,
            "identity_attack": 0.0,
        }

    monkeypatch.setattr(metrics, "_detox_predict", fake_detox)
    monkeypatch.setattr(metrics, "_textblob_sentiment", lambda text: {})
    monkeypatch.setattr(metrics, "_roberta_sentiment", lambda text: {})
    monkeypatch.setattr(metrics, "_perplexity_metrics", lambda text: {})
    monkeypatch.setattr(metrics, "_emotion_scores", lambda text: {})
    monkeypatch.setattr(
        metrics,
        "_style_metrics",
        lambda tokens, sentences, history_tokens: {
            "modal_ratio": 0.0,
            "imperative_ratio": 0.0,
            "question_ratio": 0.0,
            "jaccard_history": 0.4,
        },
    )
    monkeypatch.setattr(
        metrics,
        "_embedding_similarity",
        lambda prompt, resp, hist: {"embedding_history_similarity": 0.2},
    )

    df = compute_metrics(results)
    assert pytest.approx(0.1) == df.iloc[0]["toxicity"]
    assert pytest.approx(0.4) == df.iloc[0]["history_toxicity"]
    assert pytest.approx(-0.3) == df.iloc[0]["toxicity_delta"]
