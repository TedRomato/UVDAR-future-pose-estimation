"""Shared matplotlib style.

Loads `plot_style.yaml` (next to this file) into matplotlib's rcParams.
Call :func:`apply_style` once near the top of any plotting script:

    from plot_style import apply_style
    apply_style()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import yaml


_STYLE_PATH = Path(__file__).resolve().with_name("plot_style.yaml")
_applied = False


def load_style(path: Path | str | None = None) -> dict:
    """Return the style dict from the YAML file."""
    p = Path(path) if path is not None else _STYLE_PATH
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def apply_style(path: Path | str | None = None, force: bool = False) -> None:
    """Apply the style to ``matplotlib.rcParams``. No-op on subsequent calls
    unless ``force=True``."""
    global _applied
    if _applied and not force:
        return
    plt.rcParams.update(load_style(path))
    _applied = True
