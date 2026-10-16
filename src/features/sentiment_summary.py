"""Generates a 1-page markdown summary of the current sentiment outlook,
combining recent X sentiment trend with the latest news headlines."""
from __future__ import annotations

import datetime as dt

import pandas as pd


def _trend_label(delta: float) -> str:
    if delta > 0.05:
        return "improving"
    if delta < -0.05:
        return "deteriorating"
    return "roughly stable"


def build_sentiment_summary(ticker: str, bar_df: pd.DataFrame, news_items: list[dict]) -> str:
    recent = bar_df.tail(24)  # last ~24 bars
    prior = bar_df.tail(48).head(24)

    recent_sent = recent["sentiment_engagement_wtd"].mean()
    prior_sent = prior["sentiment_engagement_wtd"].mean() if len(prior) else recent_sent
    delta = recent_sent - prior_sent

    recent_posts = int(recent["post_count"].sum())
    price_change_pct = (recent["close"].iloc[-1] / recent["close"].iloc[0] - 1) * 100 if len(recent) > 1 else 0.0

    top_news = sorted(news_items, key=lambda x: x.get("score", 0), reverse=True)[:5]
    news_lines = "\n".join(f"- [{n['title']}]({n['url']})" for n in top_news) or "- No recent news found."

    today = dt.date.today().isoformat()

    return f"""# {ticker} — X Sentiment & Price Outlook
_Generated {today}_

## Summary
Current X sentiment for **{ticker}** is **{'positive' if recent_sent > 0 else 'negative' if recent_sent < 0 else 'neutral'}**
(engagement-weighted compound score: **{recent_sent:+.3f}**), and has been **{_trend_label(delta)}**
over the prior period (Δ = {delta:+.3f}). Post volume over the last window was **{recent_posts}** posts.
Price moved **{price_change_pct:+.2f}%** over the same window.

## Key drivers (recent news)
{news_lines}

## Read-through
- Sentiment trend: {_trend_label(delta)} ({recent_sent:+.3f} now vs {prior_sent:+.3f} prior window).
- Attention: {'elevated' if recent_posts > bar_df['post_count'].mean() * 24 else 'normal'} posting
  volume relative to the full-sample average.
- This summary is descriptive only; see `reports/evaluation_report.md` for the
  model's quantitative n+0..n+5 forecasts and backtest performance.
"""
