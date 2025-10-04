"""Template loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..models.schema import QuestionnaireTemplate


class TemplateLoadError(RuntimeError):
    """Raised when a template cannot be loaded or validated."""


def load_template(path: str | Path) -> QuestionnaireTemplate:
    """Load and validate a questionnaire template from JSON."""

    template_path = Path(path)
    if not template_path.exists():
        raise TemplateLoadError(f"Template not found: {template_path}")

    try:
        payload: Any = json.loads(template_path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - simple passthrough
        raise TemplateLoadError(f"Template is not valid JSON: {template_path}") from exc

    try:
        return QuestionnaireTemplate.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover - delegated to pydantic
        raise TemplateLoadError(f"Template failed validation: {exc}") from exc


__all__ = ["TemplateLoadError", "load_template"]
