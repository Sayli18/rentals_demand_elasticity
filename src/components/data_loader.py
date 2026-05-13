"""Data loading utilities for Phase 3+ training scripts.

Reads parquet files from `data/processed/`, with paths sourced from
`config/config.yaml → data_paths`. The repo root is resolved relative to this
module file so the loader works regardless of cwd (notebook, script, REPL).

Pairs with `features.py` — typical flow:

    from src.components.data_loader import load_split
    from src.components.features    import build_features, TARGET_COLS

    train_df   = load_split("train")
    train_feat = build_features(train_df)
    X_train    = train_feat.drop(columns=TARGET_COLS)
    y_train    = train_feat["booking_bool"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"


def _load_config() -> dict:
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def load_split(split: Literal["train", "val", "test"]) -> pd.DataFrame:
    """Load one of the time-based splits from `data/processed/`.

    Args:
        split: "train", "val", or "test"

    Returns:
        Raw split parquet. Apply `features.build_features` to transform for modelling.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of train/val/test, got {split!r}")
    cfg = _load_config()
    return pd.read_parquet(_REPO_ROOT / cfg["data_paths"][split])


def load_all() -> dict[str, pd.DataFrame]:
    """Load all three splits at once. Returns {name: DataFrame}."""
    return {name: load_split(name) for name in ("train", "val", "test")}


def load_synthetic_events() -> pd.DataFrame:
    """Load the unsplit `synthetic_demand_events.parquet`.

    Useful when full-period DGP truth is needed — e.g. Phase 5 regret computation
    against `expected_booking_prob` / `expected_cancel_prob` for any (property, date).
    """
    cfg = _load_config()
    return pd.read_parquet(_REPO_ROOT / cfg["data_paths"]["synthetic_events"])
