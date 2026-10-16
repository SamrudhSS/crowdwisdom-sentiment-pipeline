"""Charting: sentiment-vs-price overlay, rolling sentiment heatmap, and
per-horizon prediction-vs-actual scatter / error matrix."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CFG


def plot_sentiment_vs_price(df: pd.DataFrame, ticker: str, out_dir: str = CFG.charts_dir) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(13, 5))

    ax1.plot(df["timestamp"], df["close"], color="black", linewidth=1.2, label="Close price")
    ax1.set_ylabel("Price")
    ax1.set_xlabel("Time")

    ax2 = ax1.twinx()
    ax2.plot(df["timestamp"], df["sentiment_engagement_wtd_roll6"], color="tab:blue", alpha=0.7,
              linewidth=1.0, label="Sentiment (6-bar, engagement-wtd)")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax2.set_ylabel("Sentiment (-1 to 1)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title(f"{ticker}: Price vs. X Sentiment")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{ticker}_sentiment_vs_price.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prediction_matrix(eval_df: pd.DataFrame, ticker: str, out_dir: str = CFG.charts_dir) -> str:
    """Bar chart matrix: Sharpe / Sortino / MaxDD / Directional Accuracy per horizon."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("sharpe_ratio", "Sharpe Ratio"),
        ("sortino_ratio", "Sortino Ratio"),
        ("max_drawdown_pct", "Max Drawdown (%)"),
        ("directional_accuracy", "Directional Accuracy"),
    ]
    for ax, (col, title) in zip(axes.flat, metrics):
        ax.bar(eval_df["horizon"].astype(str), eval_df[col], color="tab:blue")
        ax.set_title(title)
        ax.set_xlabel("Horizon (n+h)")
        ax.axhline(0, color="gray", linewidth=0.6)

    fig.suptitle(f"{ticker}: Out-of-Sample Performance by Horizon")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{ticker}_horizon_metric_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_actual_vs_predicted(results: dict, ticker: str, out_dir: str = CFG.charts_dir) -> str:
    os.makedirs(out_dir, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (h, res) in zip(axes, sorted(results.items())):
        ax.scatter(res.y_true, res.y_pred, s=6, alpha=0.4)
        lims = [min(res.y_true.min(), res.y_pred.min()), max(res.y_true.max(), res.y_pred.max())]
        ax.plot(lims, lims, color="red", linewidth=1, linestyle="--")
        ax.set_title(f"n+{h}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    fig.suptitle(f"{ticker}: Predicted vs Actual Price by Horizon (OOS)")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{ticker}_actual_vs_predicted.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
