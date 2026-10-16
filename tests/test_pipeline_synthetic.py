"""
Synthetic end-to-end smoke test: generates fake price + sentiment bars
(no network calls), runs them through feature engineering, walk-forward
training, evaluation, and chart generation to verify the full pipeline
is wired correctly. This is what CI / a reviewer without API keys can run.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CFG
from src.features.feature_engineering import build_features
from src.models.evaluate import evaluate_all
from src.models.train import train_all_horizons
from src.viz.charts import plot_actual_vs_predicted, plot_prediction_matrix, plot_sentiment_vs_price


def make_synthetic_bar_features(n=800, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")

    # sentiment as an AR(1) process, mildly predictive of next-bar returns
    sentiment = np.zeros(n)
    for i in range(1, n):
        sentiment[i] = 0.85 * sentiment[i - 1] + rng.normal(0, 0.15)
    sentiment = np.clip(sentiment, -1, 1)

    noise = rng.normal(0, 0.004, n)
    ret = 0.002 * sentiment[:-1] + noise[1:]  # sentiment(t-1) -> return(t), a mild lead/lag signal
    ret = np.concatenate([[0], ret])
    close = 100 * np.cumprod(1 + ret)

    post_count = rng.poisson(5 + 5 * (sentiment + 1), n)

    df = pd.DataFrame(
        {
            "ticker": "SYN",
            "timestamp": ts,
            "close": close,
            "post_count": post_count,
            "sentiment_mean": sentiment,
            "sentiment_engagement_wtd": sentiment + rng.normal(0, 0.05, n),
            "follower_wtd_sentiment": sentiment + rng.normal(0, 0.05, n),
        }
    )
    return df


def test_full_pipeline_on_synthetic_data():
    bar_df = make_synthetic_bar_features()

    results = train_all_horizons(bar_df, horizons=CFG.horizons, n_folds=4)
    assert set(results.keys()) == set(CFG.horizons)
    for h, res in results.items():
        assert len(res.y_true) == len(res.y_pred) > 0

    eval_df = evaluate_all(results, bar_df)
    assert len(eval_df) == len(CFG.horizons)
    for col in ("sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "directional_accuracy", "rmse"):
        assert col in eval_df.columns
        assert eval_df[col].notna().all()

    feat_df = build_features(bar_df)
    p1 = plot_sentiment_vs_price(feat_df, "SYN", out_dir="/tmp/charts_test")
    p2 = plot_prediction_matrix(eval_df, "SYN", out_dir="/tmp/charts_test")
    p3 = plot_actual_vs_predicted(results, "SYN", out_dir="/tmp/charts_test")
    for p in (p1, p2, p3):
        assert os.path.exists(p)

    print(eval_df.to_string(index=False))
    print("Synthetic end-to-end smoke test PASSED.")


if __name__ == "__main__":
    test_full_pipeline_on_synthetic_data()
