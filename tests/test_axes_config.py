import textwrap
from pathlib import Path

import pytest

from ai_ped_red_team.axes.config import AxesConfigError, load_axes_config


def test_load_axes_config(tmp_path: Path):
    content = textwrap.dedent(
        """
        [axes.gender]
        [[axes.gender.options]]
        label = "alpha"
        STUDENT_NAME = "Alex"

        [[axes.gender.options]]
        label = "beta"
        STUDENT_NAME = "Blair"

        [axes.history]
        [[axes.history.options]]
        label = "baseline"
        HISTORY_PROMPTS = []

        [[axes.history.options]]
        label = "bias"
        HISTORY_PROMPTS = ["user: Prior conversation"]
        """
    )
    config_path = tmp_path / "axes.toml"
    config_path.write_text(content)

    config = load_axes_config(config_path)
    combinations = list(config.iter_combinations())
    assert len(combinations) == 4
    label_pairs = {
        tuple(sorted((axis, option.label) for axis, option in combo[0].items()))
        for combo in combinations
    }
    assert label_pairs == {
        (("gender", "alpha"), ("history", "baseline")),
        (("gender", "alpha"), ("history", "bias")),
        (("gender", "beta"), ("history", "baseline")),
        (("gender", "beta"), ("history", "bias")),
    }


def test_load_axes_config_missing(tmp_path: Path):
    with pytest.raises(AxesConfigError):
        load_axes_config(tmp_path / "missing.toml")
