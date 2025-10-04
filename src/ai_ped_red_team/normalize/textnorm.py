"""Text normalisation helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional

try:  # pragma: no cover - spaCy optional during tests
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

_DIRECTIVE_WORDS = {"must", "should", "may", "require", "insist"}
_CATEGORY_RULES = {
    'literacy': {'reading', 'phonics', 'literacy', 'writing'},
    'attention': {'focus', 'attention', 'adhd', 'movement'},
    'confidence': {'confidence', 'self-esteem', 'motivate'},
    'behaviour': {'behaviour', 'conduct', 'regulate'},
}
_EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
_PHONE_RE = re.compile(r'\b\d{3}[- )]?\d{3}[- ]?\d{4}\b')


@lru_cache(maxsize=1)
def _get_nlp() -> Optional["spacy.language.Language"]:
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:  # pragma: no cover - model not installed
        return None


def classify_categories(text: str) -> List[str]:
    """Classify advice categories using simple keyword rules."""

    lowered = text.lower()
    matched = [
        name
        for name, keywords in _CATEGORY_RULES.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    return matched or ["general"]


def redact_pii(text: str) -> str:
    """Redact basic PII like emails and phone numbers."""

    redacted = _EMAIL_RE.sub("<EMAIL>", text)
    redacted = _PHONE_RE.sub("<PHONE>", redacted)
    return redacted


def normalize_text(text: str) -> Dict[str, object]:
    """Normalise text by trimming, collapsing whitespace, and extracting lemmas."""

    cleaned = redact_pii(text.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    nlp = _get_nlp()
    tokens: List[str] = []
    lemmas: List[str] = []
    sentences: List[str] = []
    if nlp:
        doc = nlp(cleaned)
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc if not token.is_punct]
        sentences = [sent.text.strip() for sent in doc.sents]
    else:
        tokens = cleaned.split()
        lemmas = tokens
        sentences = [cleaned]
    return {
        "text": cleaned,
        "tokens": tokens,
        "lemmas": lemmas,
        "sentences": sentences,
        "categories": classify_categories(cleaned),
    }


def directive_ratio(text: str) -> float:
    """Compute ratio of directive words to total tokens."""

    if not text.strip():
        return 0.0
    analysis = normalize_text(text)
    tokens = analysis["tokens"]
    if not tokens:
        return 0.0
    directives = sum(1 for token in tokens if token.lower().strip(".,!") in _DIRECTIVE_WORDS)
    return directives / max(len(tokens), 1)


__all__ = ["classify_categories", "directive_ratio", "normalize_text", "redact_pii"]
