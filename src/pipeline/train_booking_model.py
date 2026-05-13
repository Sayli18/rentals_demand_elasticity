"""Phase 3a — train the booking probability model (orchestrator).

Wires together the three reusable steps:
    1. Load — `components.data_loader.load_split`
    2. Feature engineering — `components.features.build_features`
    3. Training + evaluation — `components.booking_model.train_book.{train_models,evaluate,save_models}`

Reads hyperparameters from `config/config.yaml → training.{lgbm_params, xgb_params, random_state}`.
Saves fitted models to `models/booking/{logreg,lightgbm,xgboost}.joblib`.

Calibration (isotonic) and hyperparameter tuning are deferred to follow-up
iterations once baseline numbers are in.

Run:
    uv run python -m src.pipeline.train_booking_model
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.components.booking_model.train_book import (
    evaluate,
    save_models,
    train_models,
)
from src.components.data_loader import load_split
from src.components.features import TARGET_COLS, build_features

TARGET = "booking_bool"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"
_MODEL_DIR = _REPO_ROOT / "models" / "booking"


def main() -> None:
    print("--- Phase 3a: Booking probability model ---")

    with _CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    lgbm_params = cfg["training"]["lgbm_params"]
    xgb_params = cfg["training"]["xgb_params"]
    random_state = cfg["training"]["random_state"]

    # 1. Load
    print("Loading train / val ...")
    train_df = load_split("train")
    val_df = load_split("val")

    # 2. Feature engineering
    train_feat = build_features(train_df)
    val_feat = build_features(val_df)

    X_train = train_feat.drop(columns=TARGET_COLS)
    y_train = train_feat[TARGET]
    X_val = val_feat.drop(columns=TARGET_COLS)
    y_val = val_feat[TARGET]

    print(
        f"  train: {len(X_train):>7,} rows × {X_train.shape[1]} features  "
        f"(rate {y_train.mean():.4f})"
    )
    print(
        f"  val:   {len(X_val):>7,} rows × {X_val.shape[1]} features  "
        f"(rate {y_val.mean():.4f})"
    )

    # 3. Train + evaluate
    print("\nTraining LogReg / LightGBM / XGBoost ...")
    models = train_models(X_train, y_train, lgbm_params, xgb_params, random_state)

    print("\nValidation metrics:")
    metrics = evaluate(models, X_val, y_val)
    print(metrics.to_string())

    save_models(models, _MODEL_DIR)
    print(f"\nSaved models to {_MODEL_DIR.relative_to(_REPO_ROOT)}/")


if __name__ == "__main__":
    main()
