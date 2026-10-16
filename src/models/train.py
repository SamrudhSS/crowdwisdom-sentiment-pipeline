"""
Trains one LightGBM regressor per horizon (n+0..n+5), validated with
strict walk-forward (expanding-window) splits -- never shuffled, so no
future bar ever leaks into a training fold.

Walk-forward scheme:
  fold 1: train[0:T0]            -> test[T0:T0+S]
  fold 2: train[0:T0+S]          -> test[T0+S:T0+2S]
  ...
Each fold's test predictions are concatenated into a single out-of-sample
(OOS) series per horizon, which is what evaluate.py scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import CFG
from src.features.feature_engineering import build_features, get_feature_columns


@dataclass
class FoldResult:
    horizon: int
    timestamps: pd.Series
    y_true: np.ndarray
    y_pred: np.ndarray


def walk_forward_splits(n_rows: int, n_folds: int = 5, min_train_frac: float = 0.5):
    min_train = int(n_rows * min_train_frac)
    remaining = n_rows - min_train
    fold_size = max(remaining // n_folds, 1)
    start = min_train
    while start < n_rows:
        end = min(start + fold_size, n_rows)
        yield slice(0, start), slice(start, end)
        start = end


def train_horizon(df: pd.DataFrame, feature_cols: list[str], horizon: int, n_folds: int = 5) -> FoldResult:
    target_col = f"target_t{horizon}"
    X = df[feature_cols].values
    y = df[target_col].values
    ts = df["timestamp"]

    all_ts, all_true, all_pred = [], [], []

    for train_idx, test_idx in walk_forward_splits(len(df), n_folds=n_folds):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        if len(X_test) == 0 or len(X_train) < 30:
            continue

        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        all_ts.append(ts.iloc[test_idx])
        all_true.append(y_test)
        all_pred.append(preds)

    return FoldResult(
        horizon=horizon,
        timestamps=pd.concat(all_ts).reset_index(drop=True),
        y_true=np.concatenate(all_true),
        y_pred=np.concatenate(all_pred),
    )


def train_all_horizons(bar_features_df: pd.DataFrame, horizons=CFG.horizons, n_folds: int = 5) -> dict[int, FoldResult]:
    df = build_features(bar_features_df, horizons=horizons)
    feature_cols = get_feature_columns()
    results = {}
    for h in horizons:
        results[h] = train_horizon(df, feature_cols, h, n_folds=n_folds)
    return results
