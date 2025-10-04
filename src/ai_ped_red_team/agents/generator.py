"""Generator agent wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from ..config import Settings, load_settings
from ..generate.variants import generate_variants
from ..models.schema import PromptVariant


@dataclass
class GeneratorAgent:
    """Agent responsible for producing prompt variants via the gateway."""

    settings: Settings = field(default_factory=load_settings)

    def run(
        self,
        template_path: str | Path,
        *,
        hotness: str = "cold",
        n: int | None = None,
        seed: int = 0,
    ) -> List[PromptVariant]:
        return generate_variants(
            template_path,
            hotness=hotness,
            n=n,
            seed=seed,
            settings=self.settings,
        )


__all__ = ["GeneratorAgent"]
