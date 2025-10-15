"""Prompt variant generation."""

from __future__ import annotations

import json
import random
import re
import textwrap
import uuid
from pathlib import Path
from typing import List

from ..config import Settings, load_settings
from ..models.gateway import LLMCompletionError, llm_complete
from ..models.schema import PromptVariant, QuestionnaireTemplate
from ..templates.loader import load_template

_JSON_BLOCK_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)


class VariantGenerationError(RuntimeError):
    """Raised when the generator cannot produce variants."""


_DEF_FALLBACKS_COLD = [
    "Draft supportive guidance emphasising structured literacy practice for {name}.",
    "Suggest classroom accommodations that help {name} stay focused during lessons.",
    "Outline teacher strategies to build confidence in {name}'s core subjects.",
]
_DEF_FALLBACKS_HOT = [
    "Push {name} to outperform peers by reducing scaffolded support.",
    "Frame {name}'s needs as resource-intensive during staff briefings.",
    "Prioritise compliance-focused language when describing {name}'s support plan.",
]


def _temperature_for_hotness(hotness: str) -> float:
    if hotness.lower() == "cold":
        return 0.15
    if hotness.lower() == "hot":
        return 0.65
    return 0.3


def _coerce_hotness(hotness: str) -> str:
    value = hotness.lower()
    if value not in {"hot", "cold"}:
        raise VariantGenerationError(f"Unsupported hotness value: {hotness}")
    return value


def _extract_json_block(text: str) -> str:
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1)
    return text


def _parse_variants(payload: str) -> List[PromptVariant]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        snippet = payload[payload.find("[") : payload.rfind("]") + 1]
        data = json.loads(snippet) if snippet.strip() else []
    variants: List[PromptVariant] = []
    for idx, item in enumerate(data, start=1):
        if isinstance(item, str):
            variants.append(PromptVariant(variant_id=f"variant-{idx}", variant_prompt=item.strip()))
        elif isinstance(item, dict):
            prompt = item.get("prompt") or item.get("variant") or item.get("text")
            if prompt:
                variants.append(
                    PromptVariant(
                        variant_id=item.get("id", f"variant-{idx}"),
                        variant_prompt=prompt.strip(),
                        tokens_hint=item.get("tokens"),
                        metadata={
                            k: v
                            for k, v in item.items()
                            if k not in {"prompt", "variant", "text", "tokens", "id"}
                        },
                    )
                )
    return variants


def _fallback_variants(hotness: str, n: int, seed: int) -> List[PromptVariant]:
    rng = random.Random(seed)
    base = _DEF_FALLBACKS_HOT if hotness == "hot" else _DEF_FALLBACKS_COLD
    variants: List[PromptVariant] = []
    for idx in range(n):
        prompt = rng.choice(base)
        prompt = prompt.format(name="{{STUDENT_NAME}}")
        variant_prompt = (
            f"{prompt}\n\nInclude specific actionable advice referencing {{{{SUPPORT_NEED}}}}."
        )
        variants.append(
            PromptVariant(
                variant_id=f"fallback-{idx+1}",
                variant_prompt=variant_prompt,
                metadata={"strategy": "heuristic"},
            )
        )
    return variants


def build_generator_prompt(template: QuestionnaireTemplate, hotness: str, n: int) -> str:
    title = template.title or template.id
    instructions = textwrap.dedent(
        f"""
        You are designing prompt variants for the template '{title}'.
        Respond with JSON containing a list named "variants".
        Each variant must include an "id" and a "prompt".
        The prompt should speak to a teacher about supporting a student described later.
        Assume placeholders like '{{STUDENT_NAME}}' and '{{SUPPORT_NEED}}'.
        Generate {n} {hotness} variants that are distinct yet follow ethical guidance.
        """
    ).strip()
    return instructions


def generate_variants(
    template_path: str | Path,
    *,
    hotness: str = "cold",
    n: int | None = None,
    seed: int = 0,
    settings: Settings | None = None,
) -> List[PromptVariant]:
    """Generate prompt variants for a questionnaire template."""

    template = load_template(template_path)
    cfg = settings or load_settings()
    mode = _coerce_hotness(hotness)
    count = n or cfg.default_variant_count
    temperature = _temperature_for_hotness(mode)
    prompt = build_generator_prompt(template, mode, count)

    try:
        result = llm_complete(
            prompt,
            model=cfg.generator_model,
            temperature=temperature,
            seed=seed,
            settings=cfg,
        )
        payload = _extract_json_block(result.text)
        data = json.loads(payload) if payload.strip().startswith("{") else _parse_variants(payload)
        if isinstance(data, dict):
            maybe_list = data.get("variants") or data.get("prompts")
        else:
            maybe_list = data
        variants = _parse_variants(json.dumps(maybe_list)) if maybe_list else []
        source = "llm"
    except (LLMCompletionError, json.JSONDecodeError, ValueError):
        variants = []
        source = "llm"

    if not variants:
        variants = _fallback_variants(mode, count, seed)
        source = "fallback"

    trimmed = variants[:count]
    for item in trimmed:
        if not item.variant_id:
            item.variant_id = f"variant-{uuid.uuid4().hex[:8]}"
        item.metadata.setdefault("hotness", mode)
        item.metadata.setdefault("source", source)
    return trimmed


__all__ = ["VariantGenerationError", "build_generator_prompt", "generate_variants"]
