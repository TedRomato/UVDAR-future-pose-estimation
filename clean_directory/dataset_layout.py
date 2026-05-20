"""Canonical filenames for files written by ``clean_directory.data.parse``.

Single source of truth for the filenames living inside a parsed-dataset
directory. Loaded once from :mod:`dataset_layout.yaml` next to this file,
then exposed as both a ``LAYOUT`` dict and uppercase module-level constants.

Usage::

    from clean_directory.dataset_layout import (
        BLINKERS_RIGHT,
        FLIER_ODOM_IN_CAMERA_FRAME,
        UVDAR_ESTIMATE_IN_CAMERA_FRAME,
        LAYOUT,
    )

    blinkers_path = os.path.join(run_dir, BLINKERS_RIGHT)

To add a new file: add it to ``dataset_layout.yaml`` only — this module
will pick it up automatically.
"""

from __future__ import annotations

import os
from typing import Dict

import yaml

_YAML_PATH = os.path.join(os.path.dirname(__file__), "dataset_layout.yaml")

with open(_YAML_PATH) as _f:
    LAYOUT: Dict[str, str] = yaml.safe_load(_f)

if not isinstance(LAYOUT, dict) or not all(
    isinstance(k, str) and isinstance(v, str) for k, v in LAYOUT.items()
):
    raise RuntimeError(
        f"{_YAML_PATH}: must be a flat mapping of str -> str (filename).")

# Expose every entry as an uppercase module-level constant so call sites
# can do ``from clean_directory.dataset_layout import BLINKERS_RIGHT``.
globals().update({k.upper(): v for k, v in LAYOUT.items()})

__all__ = ["LAYOUT", *(k.upper() for k in LAYOUT)]
