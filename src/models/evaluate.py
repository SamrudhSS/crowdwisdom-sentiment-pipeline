"""
Evaluates each horizon's out-of-sample predictions two ways:
  1. Pure forecasting error: RMSE, MAE, MAPE, directional accuracy.
  2. A simple long/flat trading strategy derived from the prediction
     (go long the asset from t to t+h when predicted return > 0, else
     stay flat), scored with Sharpe, Sortino and Max Drawdown -- the
     three metrics requested in the brief.

Annualization assumes the configured bar size (1h -> sqrt(24*252) scaling,
1d -> sqrt(252)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG
from src.models.train import FoldResult

_BARS_PER_YEAR = {"1h": 24 * 252, "1d": 252}


def _ann_factor(bar_size: str = CFG.bar_size) -> float:
    return np.sqrt(_BARS_PER_YEAR.get(bar_size, 252))


def forecast_errors(result: FoldResult, last_close: np.ndarray) -> dict:
    y_true, y_pred = result.y_true, result.y_pred
    err = y_pred - y_true
    mape = np.mean(np.abs(err) / np.clip(np.abs(y_true), 1e-9, None)) * 100

    true_dir = np.sign(y_true - last_close)
    pred_dir = np.sign(y_pred - last_close)
    dir_acc = np.mean(true_dir == pred_dir)

    return {
        "horizon": result.horizon,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape_pct": float(mape),
        "directional_accuracy": float(dir_acc),
        "n_obs": int(len(y_true)),
    }


def strategy_returns(result: FoldResult, last_close: np.ndarray) -> np.ndarray:
    """Return of a long/flat rule: long if predicted move is positive."""
    predicted_move = result.y_pred - last_close
    actual_return = (result.y_true - last_close) / last_close
    position = (predicted_move > 0).astype(float)
    return position * actual_return


def sharpe_ratio(returns: np.ndarray, bar_size: str = CFG.bar_size) -> float:
    if returns.std(ddof=1) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * _ann_factor(bar_size))


def sortino_ratio(returns: np.ndarray, bar_size: str = CFG.bar_size) -> float:
    downside = returns[returns < 0]
    dd_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if dd_std == 0:
        return 0.0
    return float(returns.mean() / dd_std * _ann_factor(bar_size))


def max_drawdown(returns: np.ndarray) -> float:
    equity = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def evaluate_all(results: dict[int, FoldResult], bar_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    bar_features_df must contain 'timestamp' and 'close' so we can look up
    the close price at the time each OOS prediction was made (n+0 close).
    """
    close_by_ts = bar_features_df.set_index("timestamp")["close"]
    rows = []
    for h, result in results.items():
        last_close = close_by_ts.reindex(result.timestamps).values
        err_stats = forecast_errors(result, last_close)
        rets = strategy_returns(result, last_close)
        err_stats.update(
            {
                "sharpe_ratio": sharpe_ratio(rets),
                "sortino_ratio": sortino_ratio(rets),
                "max_drawdown_pct": max_drawdown(rets) * 100,
                "strategy_total_return_pct": (np.prod(1 + rets) - 1) * 100,
            }
        )
        rows.append(err_stats)
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
