"""Feature engineering for Phase 3 booking + cancellation models.

This module does ONE job: turn a raw split parquet (train / val / test) into a
feature-ready dataframe by

  1. dropping columns that shouldn't be features (leakage, identifying-noise,
     synthetic-only artifacts, identifiers)
  2. (creating any derived features — none today; new ones go inside the function)
  3. one-hot encoding the seven property-categorical columns

It does NOT split X from y — that's the training script's job. Target columns
(`booking_bool`, `cancelled_bool`) are passed through untouched so the caller
can pick whichever target they need.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# DGP truth — never feed to a model (these literally encode the answer)
LEAK_COLS = [
    "expected_booking_prob",
    "expected_cancel_prob",
    "beta_base",
    "beta_discount",
]

# DML identifying variation — must stay out of W (would absorb residual variation
# in date_base_price that identifies β_base)
IDENT_NOISE_COLS = ["host_price_shock"]

# Synthetic-only artifacts — no real-world analog (host doesn't separately
# configure base_price / seasonal_mult / weekend_mult on a real platform)
ARTIFACT_COLS = ["base_price", "seasonal_mult", "weekend_mult"]

# High-cardinality identifiers / already decomposed
ID_COLS = ["property_id", "stay_date"]

# Categorical features to one-hot encode
CATEGORICAL_COLS = [
    "region",
    "country",
    "quality_tier",
    "host_pricing_style",
    "property_type",
    "view",
    "cancel_policy",
]

# Targets — kept in the output; caller drops these when building X
TARGET_COLS = ["booking_bool", "cancelled_bool"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a raw split parquet into a feature-ready dataframe.

    Steps: drop leakage/artifact/identifier columns → (derive new features) →
    one-hot encode categoricals.

    Targets are NOT removed — the caller separates them when building (X, y).

    Args:
        df: rows from `data/processed/{train,val,test}.parquet`

    Returns:
        Feature-ready dataframe with one-hot-encoded categoricals.
        Both `booking_bool` and `cancelled_bool` are still present.
    """
    drop_cols = LEAK_COLS + IDENT_NOISE_COLS + ARTIFACT_COLS + ID_COLS
    out = df.drop(columns=drop_cols, errors="ignore")

    # --- Hook for derived features ---
    # Add new features here as needed, e.g.:
    out["log_date_base_price"] = np.log(out["date_base_price"])
    #   out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    #   out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["is_summer"] = out["month"].isin([6, 7, 8]).astype(int)

    out = pd.get_dummies(out, columns=CATEGORICAL_COLS, dtype=int)

    return out
