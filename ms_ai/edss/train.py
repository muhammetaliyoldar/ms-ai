"""
EDSS regression training with patient-level GroupKFold cross-validation.

Uses RandomForestRegressor and reports per-fold MAE.
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

log = logging.getLogger("ms_ai.train_edss")

# ── Feature engineering ────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "lesion_volume_mm3",
    "lesion_count",
    "mean_lesion_size_mm3",
    "max_lesion_size_mm3",
    "age",
]

SEX_MAP = {"M": 0, "F": 1}


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the feature matrix from the merged DataFrame.

    Returns (X DataFrame, feature_names list).
    """
    X = df[NUMERIC_FEATURES].copy()

    # sex → numeric
    X["sex"] = df["sex"].map(SEX_MAP).fillna(0).astype(int)

    # ms_type → one-hot  (drop_first=False so every type gets a column)
    if "ms_type" in df.columns:
        dummies = pd.get_dummies(df["ms_type"], prefix="ms_type")
        # Ensure consistent column ordering
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        X = pd.concat([X, dummies], axis=1)

    feature_names = list(X.columns)
    return X, feature_names


# ── Cross-validation ──────────────────────────────────────────────────────

def cross_validate(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> Tuple[List[float], float, float]:
    """
    Patient-level GroupKFold CV with RandomForestRegressor.

    Returns (per_fold_maes, mean_mae, std_mae).
    """
    X, feature_names = prepare_features(df)
    y = df["edss"].values
    groups = df["patient"].values

    gkf = GroupKFold(n_splits=n_splits)
    fold_maes: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        model = RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X.iloc[train_idx], y[train_idx])
        preds = model.predict(X.iloc[val_idx])
        mae = mean_absolute_error(y[val_idx], preds)
        fold_maes.append(round(mae, 4))
        log.info("  Fold %d: MAE = %.4f  (val_size=%d)", fold_idx, mae, len(val_idx))

    mean_mae = float(np.mean(fold_maes))
    std_mae = float(np.std(fold_maes))
    return fold_maes, mean_mae, std_mae


# ── Final model ───────────────────────────────────────────────────────────

def train_final_model(
    df: pd.DataFrame,
    mask_source: str,
    split: str,
    artifacts_dir: Path,
) -> Path:
    """
    Fit a final RF on all rows with non-null EDSS, save model + metadata.

    Returns path to the saved model pickle.
    """
    X, feature_names = prepare_features(df)
    y = df["edss"].values

    model = RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)

    # Save model
    model_path = artifacts_dir / f"edss_model_{mask_source}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata
    meta: Dict = {
        "features": feature_names,
        "n_samples": len(df),
        "mask_source": mask_source,
        "split": split,
        "n_estimators": 600,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = artifacts_dir / f"edss_model_{mask_source}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model_path
