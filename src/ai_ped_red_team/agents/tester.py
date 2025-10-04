"""Tester agent wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..config import Settings, load_settings
from ..models.schema import PromptVariant
from ..run.runner import RunArtifacts, RunExecutionConfig, run_variants


@dataclass
class TesterAgent:
    """Agent responsible for executing prompt variants against EHCP profiles."""

    settings: Settings = field(default_factory=load_settings)

    def run(
        self,
        template_path: str | Path,
        ehcp_dir: str | Path,
        *,
        config: Optional[RunExecutionConfig] = None,
        variants: Optional[Iterable[PromptVariant]] = None,
    ) -> RunArtifacts:
        return run_variants(
            template_path,
            ehcp_dir,
            config=config,
            settings=self.settings,
            variants=variants,
        )


__all__ = ["TesterAgent"]
