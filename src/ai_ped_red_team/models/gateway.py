"""LLM gateway built on LiteLLM."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "timeout": timeout or cfg.request_timeout,
            }
            if seed is not None:
                call_kwargs["seed"] = seed
            response = completion(**call_kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000
            text = _extract_text(response)
            metadata = {
                "usage": response.get("usage", {}),
                "id": response.get("id"),
                "provider": response.get("provider"),
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
