"""Axis configuration loader for templated EHCP experiments."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class AxisOption:
    """Single option for an axis (e.g., gender, support, history)."""

    label: str
    values: Mapping[str, Any]


@dataclass(frozen=True)
class AxesConfig:
    """Collection of axes and their available options."""

    axes: Mapping[str, Tuple[AxisOption, ...]]

    def iter_combinations(self) -> Iterator[Tuple[Mapping[str, AxisOption], Mapping[str, Any]]]:
        """Yield cartesian product of axis options and merged values."""

        if not self.axes:
            yield {}, {}
            return

        axis_names = list(self.axes.keys())
        axis_option_lists = [self.axes[name] for name in axis_names]

        for option_tuple in itertools.product(*axis_option_lists):
            axis_mapping = {
                name: option for name, option in zip(axis_names, option_tuple, strict=False)
            }
            merged: Dict[str, Any] = {}
            for option in option_tuple:
                for key, value in option.values.items():
                    merged[key] = value
            yield axis_mapping, merged


class AxesConfigError(RuntimeError):
    """Raised when an axes configuration cannot be parsed."""


_SUPPORTED_SPECIAL_KEYS = {"HISTORY_PROMPTS"}


def _parse_axis_options(raw_options: Iterable[Mapping[str, Any]]) -> Tuple[AxisOption, ...]:
    options: List[AxisOption] = []
    for payload in raw_options:
        if "label" not in payload:
            raise AxesConfigError("Axis option missing 'label'.")
        label = str(payload["label"]).strip()
        values = {k: v for k, v in payload.items() if k != "label"}
        options.append(AxisOption(label=label, values=values))
    return tuple(options)


def load_axes_config(path: str | Path) -> AxesConfig:
    """Load axes configuration from a TOML file."""

    config_path = Path(path)
    if not config_path.exists():
        raise AxesConfigError(f"Axes config not found: {config_path}")
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    raw_axes = data.get("axes")
    if not raw_axes:
        raise AxesConfigError("No axes defined in configuration.")

    axes: Dict[str, Tuple[AxisOption, ...]] = {}
    for axis_name, axis_payload in raw_axes.items():
        options_key = "options"
        raw_options = axis_payload.get(options_key)
        if raw_options is None:
            raise AxesConfigError(f"Axis '{axis_name}' missing '{options_key}'.")
        options = _parse_axis_options(raw_options)
        axes[axis_name] = options
    return AxesConfig(axes=axes)


__all__ = ["AxisOption", "AxesConfig", "AxesConfigError", "load_axes_config"]
