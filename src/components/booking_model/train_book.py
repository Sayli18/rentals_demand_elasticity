"""Phase 3a — booking probability model: training & evaluation primitives.

Pure functions only. Orchestration (loading data, building features, calling
these in sequence) lives in `src/pipeline/train_booking_model.py`.

Functions:
    train_models — fit LogReg, LightGBM, XGBoost on (X, y)
    evaluate     — AUC, Brier, log-loss on val
    save_models  — joblib dump under a given directory
"""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lgbm_params: dict,
    xgb_params: dict,
    random_state: int,
) -> dict:
    """Fit LogReg, LightGBM, XGBoost on the same training data.

    Returns:
        dict mapping {"logreg", "lightgbm", "xgboost"} → fitted model
    """
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, n_jobs=-1, random_state=random_state)),
    ])
    logreg.fit(X_train, y_train)

    lgbm = lgb.LGBMClassifier(
        **lgbm_params, random_state=random_state, n_jobs=-1, verbose=-1
    )
    lgbm.fit(X_train, y_train)

    xgbm = xgb.XGBClassifier(
        **xgb_params, random_state=random_state, n_jobs=-1,
        eval_metric="logloss", verbosity=0,
    )
    xgbm.fit(X_train, y_train)

    return {"logreg": logreg, "lightgbm": lgbm, "xgboost": xgbm}


def evaluate(models: dict, X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
    """Compute AUC, Brier, log-loss for each model on val."""
    rows = []
    for name, model in models.items():
        y_hat = model.predict_proba(X_val)[:, 1]
        rows.append({
            "model": name,
            "auc": roc_auc_score(y_val, y_hat),
            "brier": brier_score_loss(y_val, y_hat),
            "log_loss": log_loss(y_val, y_hat),
        })
    return pd.DataFrame(rows).set_index("model").round(4)


def save_models(models: dict, model_dir: Path) -> None:
    """Joblib-dump each model under model_dir as `<name>.joblib`."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, model_dir / f"{name}.joblib")
