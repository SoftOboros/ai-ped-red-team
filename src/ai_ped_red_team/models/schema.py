"""Pydantic schemas used across the project."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QuestionnaireTemplate(BaseModel):
    """Template definition for questionnaire-driven prompts."""

    id: str
    title: Optional[str] = None
    hotness: str = Field(default="cold", pattern="^(cold|hot)$")
    template_text: str = Field(default="{{prompt}}")
    placeholders: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PromptVariant(BaseModel):
    """Represents a single prompt variant generated from a template."""

    variant_id: str
    variant_prompt: str
    tokens_hint: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """Configuration used for executing a run."""

    model: str
    temperature: float = 0.0
    seed: Optional[int] = None
    n_variants: int = 5
    counterbalance: bool = True
    extra: Dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Result of running a variant against a specific EHCP profile."""

    variant_id: str
    student: str
    prompt: str
    response: str
    latency_ms: float
    model_info: Dict[str, Any]
    started_at: datetime
    completed_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisRecord(BaseModel):
    """Aggregated metrics for a single response."""

    variant_id: str
    student: str
    word_count: int
    char_count: int
    directive_ratio: float
    readability: Optional[float] = None
    sentiment: Optional[float] = None
    toxicity: Optional[float] = None
    severe_toxicity: Optional[float] = None
    obscene: Optional[float] = None
    threat: Optional[float] = None
    insult: Optional[float] = None
    identity_attack: Optional[float] = None
    history_toxicity: Optional[float] = None
    history_severe_toxicity: Optional[float] = None
    history_obscene: Optional[float] = None
    history_threat: Optional[float] = None
    history_insult: Optional[float] = None
    history_identity_attack: Optional[float] = None
    toxicity_delta: Optional[float] = None
    severe_toxicity_delta: Optional[float] = None
    obscene_delta: Optional[float] = None
    threat_delta: Optional[float] = None
    insult_delta: Optional[float] = None
    identity_attack_delta: Optional[float] = None
    textblob_polarity: Optional[float] = None
    textblob_subjectivity: Optional[float] = None
    roberta_label: Optional[str] = None
    roberta_score: Optional[float] = None
    perplexity: Optional[float] = None
    per_token_loss: Optional[float] = None
    emotion_label: Optional[str] = None
    emotion_score: Optional[float] = None
    modal_ratio: Optional[float] = None
    imperative_ratio: Optional[float] = None
    question_ratio: Optional[float] = None
    jaccard_history: Optional[float] = None
    embedding_prompt_similarity: Optional[float] = None
    embedding_history_similarity: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "QuestionnaireTemplate",
    "PromptVariant",
    "RunConfig",
    "RunResult",
    "AnalysisRecord",
]
