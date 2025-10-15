"""LLM gateway built on LiteLLM."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    ModelExceptionRule(
        vendor="gemini",
        model_prefixes=(
            "gemini-",
            "learnlm-",
            "gemma-",
            "imagen-",
            "embedding-",
            "text-embedding-",
        ),
        drop_params=("seed",),
    ),
    ModelExceptionRule(
        vendor="anthropic",
        model_prefixes=("claude-", "claude2", "haiku-", "sonnet-"),
        drop_params=("seed",),
    ),
    ModelExceptionRule(
        vendor="cohere",
        model_prefixes=("command-", "cohere-", "xlarge", "medium", "small", "embed-", "rerank-"),
        drop_params=("seed",),
    ),
)

_VENDOR_PREFIX_HINTS: Dict[str, Tuple[str, ...]] = {
    "openai": ("gpt-", "o1-", "o3-", "text-", "whisper-", "chatgpt-"),
    "anthropic": ("claude-", "haiku-", "sonnet-", "opus-"),
    "mistral": ("mistral-", "mixtral-", "open-mixtral"),
    "cohere": ("command-", "cohere-", "embed-", "rerank-"),
    "gemini": (
        "gemini-",
        "learnlm-",
        "gemma-",
        "imagen-",
        "embedding-",
        "text-embedding-",
    ),
}

_CREDENTIAL_FIELDS: Dict[str, Tuple[str, str]] = {
    "openai": ("openai_api_key", "OPENAI_API_KEY"),
    "gemini": ("google_api_key", "GOOGLE_API_KEY"),
    "google": ("google_api_key", "GOOGLE_API_KEY"),
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    "openrouter": ("openrouter_api_key", "OPENROUTER_API_KEY"),
    "mistral": ("mistral_api_key", "MISTRAL_API_KEY"),
    "cohere": ("cohere_api_key", "COHERE_API_KEY"),
}


def _split_model_name(model_name: str) -> Tuple[str, str]:
    if "/" in model_name:
        vendor, _, remainder = model_name.partition("/")
        return vendor, remainder
    return "", model_name


def _infer_vendor_hint(model_name: str) -> Optional[str]:
    for vendor, prefixes in _VENDOR_PREFIX_HINTS.items():
        for prefix in prefixes:
            if model_name.startswith(prefix):
                return vendor
    return None


def _normalize_model_name(raw_name: str) -> str:
    if "/" in raw_name or not raw_name:
        return raw_name
    vendor = _infer_vendor_hint(raw_name)
    if vendor:
        return f"{vendor}/{raw_name}"
    return raw_name


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


def _massage_messages_for_vendor(
    vendor: str, messages: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    adjusted = [dict(item) for item in messages]
    if vendor != "cohere" or len(adjusted) < 2:
        return adjusted
    final_role = (adjusted[-1].get("role") or "").lower()
    prior_role = (adjusted[-2].get("role") or "").lower()
    if final_role == "user" and prior_role == "user":
        adjusted.insert(-1, {"role": "assistant", "content": " "})
    return adjusted


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
    model_name = _normalize_model_name(model or cfg.tester_model)
    vendor, _ = _split_model_name(model_name)
    cred = _CREDENTIAL_FIELDS.get(vendor)
    if cred:
        attr, env_name = cred
        if not getattr(cfg, attr, None):
            raise LLMCompletionError(
                f"Missing credentials for vendor '{vendor}'. Set {env_name} before continuing."
            )
    delay = cfg.backoff_seconds
    attempts = cfg.max_retries + 1

    last_exc: Optional[Exception] = None
    start_time = time.perf_counter()

    for attempt in range(1, attempts + 1):
        try:
            call_kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": (
                    _massage_messages_for_vendor(vendor, messages)
                    if messages is not None
                    else [{"role": "user", "content": prompt}]
                ),
                "temperature": temperature,
                "timeout": timeout or cfg.request_timeout,
            }
            if seed is not None:
                call_kwargs["seed"] = seed
            litellm_params = call_kwargs.get("litellm_params")
            if isinstance(litellm_params, dict):
                litellm_params.setdefault("drop_params", True)
            else:
                call_kwargs["litellm_params"] = {"drop_params": True}
            if vendor:
                litellm_params = call_kwargs["litellm_params"]
                litellm_params["custom_llm_provider"] = vendor
                if vendor == "mistral" and "api_base" not in litellm_params:
                    litellm_params["api_base"] = "https://api.mistral.ai/v1"
                call_kwargs.setdefault("custom_llm_provider", vendor)
            if vendor == "cohere":
                # Cohere chat defaults to force_single_step=True; disable unless
                # explicitly requested.
                call_kwargs.setdefault("force_single_step", False)
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

    detail = str(last_exc) if last_exc is not None else "unknown error"
    raise LLMCompletionError(
        f"LLM completion failed after {attempts} attempts; last error: {detail}"
    ) from last_exc


__all__ = ["LLMCompletionError", "LLMResult", "llm_complete"]
