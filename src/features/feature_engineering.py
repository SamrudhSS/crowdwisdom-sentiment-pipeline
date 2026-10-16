"""
Builds the model-ready feature matrix and multi-horizon targets from
bar_features (price + sentiment only, as required by the spec).

Features (all computable at time t, no lookahead):
  - price: ret_1, ret_3, ret_6, ret_12 (lagged returns over k bars)
  - price: rolling volatility (std of ret_1 over 6/12 bars)
  - sentiment: level (mean / engagement-wtd / follower-wtd) at t
  - sentiment: rolling mean over 3/6/12 bars (smoothed sentiment)
  - sentiment: momentum = sentiment(t) - sentiment(t-3)
  - sentiment: post_count and its rolling mean (attention/volume proxy)
  - time: hour-of-day, day-of-week (cyclical encoding)

Targets: close price at t+0..t+5 bars (n = current bar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG

FEATURE_COLUMNS: list[str] = []  # populated by build_features()


def _cyclical(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    radians = 2 * np.pi * series / period
    return np.sin(radians), np.cos(radians)


def build_features(df: pd.DataFrame, horizons=CFG.horizons) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    # fill sentiment gaps (bars with no posts) with 0 (neutral) / forward-fill
    sent_cols = ["sentiment_mean", "sentiment_engagement_wtd", "follower_wtd_sentiment"]
    for c in sent_cols:
        df[c] = df[c].ffill().fillna(0.0)
    df["post_count"] = df["post_count"].fillna(0)

    # --- price features ---
    df["ret_1"] = df["close"].pct_change(1)
    for k in (3, 6, 12):
        df[f"ret_{k}"] = df["close"].pct_change(k)
    df["vol_6"] = df["ret_1"].rolling(6).std()
    df["vol_12"] = df["ret_1"].rolling(12).std()

    # --- sentiment features ---
    for c in sent_cols:
        for w in (3, 6, 12):
            df[f"{c}_roll{w}"] = df[c].rolling(w).mean()
        df[f"{c}_momentum3"] = df[c] - df[c].shift(3)

    df["post_count_roll6"] = df["post_count"].rolling(6).mean()
    df["post_count_z"] = (df["post_count"] - df["post_count"].rolling(24).mean()) / (
        df["post_count"].rolling(24).std() + 1e-9
    )

    # --- time features ---
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["hour_sin"], df["hour_cos"] = _cyclical(df["hour"], 24)
    df["dow_sin"], df["dow_cos"] = _cyclical(df["dow"], 7)

    # --- targets: price at t+0..t+5 ---
    for h in horizons:
        df[f"target_t{h}"] = df["close"].shift(-h)

    global FEATURE_COLUMNS
    FEATURE_COLUMNS = [
        c for c in df.columns
        if c not in ("ticker", "timestamp", "close", "hour", "dow")
        and not c.startswith("target_t")
    ]

    # drop rows without full feature history or full target horizon
    df = df.dropna(subset=FEATURE_COLUMNS + [f"target_t{h}" for h in horizons]).reset_index(drop=True)
    return df


def get_feature_columns() -> list[str]:
    return FEATURE_COLUMNS
