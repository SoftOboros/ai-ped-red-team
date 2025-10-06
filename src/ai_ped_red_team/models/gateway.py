"""LLM gateway built on LiteLLM."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Sequence

from litellm import completion

from ..config import Settings, load_settings


class LLMCompletionError(RuntimeError):
    """Raised when an LLM completion fails after retries."""


@dataclass
class LLMResult:
    """Return object for LLM calls."""

    text: str
    latency_ms: float
    model: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ModelExceptionRule:
    """Rule capturing vendor/model quirks that require parameter overrides."""

    vendor: str
    model_prefixes: Tuple[str, ...]
    force_params: Dict[str, Any] = field(default_factory=dict)
    drop_params: Tuple[str, ...] = field(default_factory=tuple)

    def matches(self, vendor: str, model_name: str) -> bool:
        if self.vendor and vendor != self.vendor:
            return False
        return any(model_name.startswith(prefix) for prefix in self.model_prefixes)


_MODEL_EXCEPTION_TABLE: Tuple[ModelExceptionRule, ...] = (
    ModelExceptionRule(
        vendor="openai",
        model_prefixes=("gpt-5",),
        force_params={"temperature": 1.0},
    ),
)


def _split_model_name(model_name: str) -> Tuple[str, str]:
    if "/" in model_name:
        vendor, _, remainder = model_name.partition("/")
        return vendor, remainder
    return "", model_name


def _apply_model_overrides(model_name: str, call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    applied = {"forced": {}, "dropped": []}
    vendor, simple_name = _split_model_name(model_name)
    for rule in _MODEL_EXCEPTION_TABLE:
        if not rule.matches(vendor, simple_name):
            continue
        for key, value in rule.force_params.items():
            if call_kwargs.get(key) != value:
                call_kwargs[key] = value
                applied["forced"][key] = value
        for key in rule.drop_params:
            if key in call_kwargs:
                call_kwargs.pop(key, None)
                applied["dropped"].append(key)
    return applied


def _extract_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content", "")


def llm_complete(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    settings: Optional[Settings] = None,
    timeout: Optional[float] = None,
    messages: Optional[Sequence[Dict[str, str]]] = None,
) -> LLMResult:
    """Execute a completion request with retries and backoff."""

    cfg = settings or load_settings()
    model_name = model or cfg.tester_model
    delay = cfg.backoff_seconds
    attempts = cfg.max_retries + 1

    last_exc: Optional[Exception] = None
    start_time = time.perf_counter()

    for attempt in range(1, attempts + 1):
        try:
            call_kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": list(messages)
                if messages is not None
                else [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "timeout": timeout or cfg.request_timeout,
            }
            if seed is not None:
                call_kwargs["seed"] = seed
            override_info = _apply_model_overrides(model_name, call_kwargs)
            response = completion(**call_kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000
            text = _extract_text(response)
            metadata = {
                "usage": response.get("usage", {}),
                "id": response.get("id"),
                "provider": response.get("provider"),
            }
            if override_info["forced"] or override_info["dropped"]:
                metadata["overrides"] = {
                    "forced": dict(override_info["forced"]),
                    "dropped": list(override_info["dropped"]),
                }
            return LLMResult(text=text, latency_ms=latency_ms, model=model_name, metadata=metadata)
        except Exception as exc:  # pragma: no cover - litellm raises many types
            last_exc = exc
            if attempt >= attempts:
                break
            time.sleep(delay)
            delay *= 2

    raise LLMCompletionError(f"LLM completion failed after {attempts} attempts") from last_exc


__all__ = ["LLMCompletionError", "LLMResult", "llm_complete"]
