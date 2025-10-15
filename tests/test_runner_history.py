import json
from pathlib import Path
from typing import List

import pytest

from ai_ped_red_team.models.gateway import LLMResult
from ai_ped_red_team.models.schema import PromptVariant
from ai_ped_red_team.run.runner import RunExecutionConfig, run_variants


class DummyLLM:
    def __init__(self) -> None:
        self.calls: List[dict] = []

    def __call__(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return LLMResult(text="ok", latency_ms=1.0, model="dummy", metadata={})


@pytest.fixture
def temp_template(tmp_path: Path) -> Path:
    template_data = {
        "id": "demo",
        "template_text": "{{prompt}}",
        "placeholders": ["STUDENT_NAME", "SUPPORT_NEED"],
    }
    tpl = tmp_path / "template.json"
    tpl.write_text(json.dumps(template_data))
    return tpl


@pytest.fixture
def temp_ehcp(tmp_path: Path) -> Path:
    directory = tmp_path / "ehcp"
    directory.mkdir()
    (directory / "profile.txt").write_text("Full Name Test Student\nPrimary Need: Focus\n")
    return directory


def test_run_variants_with_history(monkeypatch, temp_template: Path, temp_ehcp: Path):
    dummy = DummyLLM()
    monkeypatch.setattr("ai_ped_red_team.run.runner.llm_complete", dummy)

    variant = PromptVariant(
        variant_id="v1",
        variant_prompt="Help {{STUDENT_NAME}} with {{SUPPORT_NEED}}",
    )
    history = [
        "user: Previous plan for {{STUDENT_NAME}}.",
        "assistant: Provided minimal guidance.",
    ]

    artefacts = run_variants(
        temp_template,
        temp_ehcp,
        config=RunExecutionConfig(model="dummy", counterbalance=False, n_variants=1),
        variants=[variant],
        substitutions={"STUDENT_NAME": "Jordan", "SUPPORT_NEED": "Focus"},
        history_prompts=history,
        axes_labels={"history": "test"},
        run_tag="test-history",
    )

    assert artefacts.variants[0].variant_id == "v1"
    assert dummy.calls, "LLM should have been invoked"
    call = dummy.calls[0]
    messages = call["messages"]
    assert messages[0]["content"].startswith("Previous plan for Jordan")
    assert messages[-1]["content"].startswith("Help Jordan")
    assert call["model"] == "dummy"

    persisted = artefacts.results_path.read_text().splitlines()
    assert persisted, "results.jsonl should contain entries"
    record = json.loads(persisted[0])
    history_meta = record["metadata"].get("history_messages")
    assert history_meta[0].startswith("user: Previous plan for Jordan")
