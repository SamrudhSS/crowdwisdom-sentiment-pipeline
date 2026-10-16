"""
End-to-end orchestrator:
  1. Scrape X posts (Apify) + fetch price (yfinance) + fetch news (Tavily/Exa)
  2. Score sentiment, store everything in DuckDB
  3. Build bar-level joined features
  4. Engineer ML features + walk-forward train 6 horizon models
  5. Evaluate (Sharpe/Sortino/MaxDD) + render charts
  6. Write reports/evaluation_report.md, reports/sentiment_summary.md,
     reports/run_manifest.json (for reproducibility)

Usage:
    python -m src.pipeline --ticker TSLA --cashtag "$TSLA" --company "Tesla"
"""
from __future__ import annotations

import argparse
import json
import os
import time

from config import CFG
from src.logging_setup import setup_logging
from src.db import database as db
from src.features.feature_engineering import build_features
from src.features.sentiment_scoring import score_posts
from src.features.sentiment_summary import build_sentiment_summary
from src.ingestion.apify_scraper import scrape_x_posts
from src.ingestion.news_fetcher import fetch_asset_news
from src.ingestion.price_fetcher import fetch_price_history
from src.models.evaluate import evaluate_all
from src.models.train import train_all_horizons
from src.viz.charts import plot_actual_vs_predicted, plot_prediction_matrix, plot_sentiment_vs_price

import pandas as pd


def run(ticker: str, cashtag: str | None, company: str | None, lookback_days: int, keywords=None):
    log = setup_logging(run_name=f"pipeline_{ticker}")
    t_start = time.time()
    os.makedirs(CFG.reports_dir, exist_ok=True)
    os.makedirs(CFG.charts_dir, exist_ok=True)
    os.makedirs(os.path.dirname(CFG.db_path) or ".", exist_ok=True)

    log.info(f"[1/6] Scraping X posts for {ticker} ({lookback_days}d, cashtag={cashtag}, keywords={keywords})")
    t0 = time.time()
    try:
        raw_posts = scrape_x_posts(ticker, cashtag=cashtag, keywords=keywords, lookback_days=lookback_days)
    except Exception:
        log.exception("X scraping failed")
        raise
    posts_df = pd.DataFrame(raw_posts)
    if not posts_df.empty:
        posts_df = score_posts(posts_df)
        posts_df["created_at_utc"] = pd.to_datetime(
    posts_df["created_at_utc"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce"
)
    log.info(f"  -> {len(posts_df)} posts scraped in {time.time()-t0:.1f}s")

    log.info("[2/6] Fetching price history + news...")
    t0 = time.time()
    try:
        price_df = fetch_price_history(ticker, lookback_days=lookback_days)
    except Exception:
        log.exception("Price fetch failed")
        raise
    try:
        news_items = fetch_asset_news(ticker, company_name=company)
    except Exception:
        log.exception("News fetch failed (continuing with empty news list)")
        news_items = []
    log.info(f"  -> {len(price_df)} price bars, {len(news_items)} news items in {time.time()-t0:.1f}s")

    con = db.get_connection()
    if not posts_df.empty:
        db.save_posts(con, posts_df)
    db.save_price_bars(con, price_df)
    if news_items:
        db.save_news(con, pd.DataFrame(news_items).assign(ticker=ticker)[
            ["ticker", "title", "url", "content", "published", "score", "source"]
        ])
    log.info("  -> saved posts/price/news to DuckDB warehouse at " + CFG.db_path)

    log.info("[3/6] Building bar-level sentiment+price features...")
    bar_df = db.build_bar_features(con, ticker)
    log.info(f"  -> {len(bar_df)} joined bars")

    log.info("[4/6] Training walk-forward models for n+0..n+5...")
    t0 = time.time()
    results = train_all_horizons(bar_df, horizons=CFG.horizons, n_folds=5)
    for h, res in results.items():
        log.info(f"  -> horizon n+{h}: {len(res.y_true)} OOS predictions")
    log.info(f"  -> training completed in {time.time()-t0:.1f}s")

    log.info("[5/6] Evaluating + charting...")
    eval_df = evaluate_all(results, bar_df)
    log.info("\n" + eval_df.to_string(index=False))
    chart_paths = [
        plot_sentiment_vs_price(build_features(bar_df), ticker),
        plot_prediction_matrix(eval_df, ticker),
        plot_actual_vs_predicted(results, ticker),
    ]
    log.info(f"  -> charts written: {chart_paths}")

    log.info("[6/6] Writing reports...")
    eval_md = eval_df.to_markdown(index=False)
    with open(os.path.join(CFG.reports_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Report — {ticker}\n\n")
        f.write("Walk-forward, out-of-sample metrics per horizon (n = current bar).\n\n")
        f.write(eval_md + "\n\n")
        f.write("## Charts\n")
        for p in chart_paths:
            f.write(f"- `{p}`\n")

    summary_md = build_sentiment_summary(ticker, bar_df, news_items)
    with open(os.path.join(CFG.reports_dir, "sentiment_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md)

    manifest = {
        "ticker": ticker,
        "cashtag": cashtag,
        "keywords": keywords,
        "lookback_days": lookback_days,
        "bar_size": CFG.bar_size,
        "apify_actor_id": CFG.apify_x_actor_id,
        "n_posts_scraped": len(posts_df),
        "n_price_bars": len(price_df),
        "n_news_items": len(news_items),
        "horizons": list(CFG.horizons),
    }
    with open(os.path.join(CFG.reports_dir, "run_manifest.json"), "w",encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    log.info(f"Done in {time.time()-t_start:.1f}s total. See reports/, charts/, and logs/.")
    return eval_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--cashtag", default=None)
    parser.add_argument("--company", default=None)
    parser.add_argument("--days", type=int, default=CFG.lookback_days)
    parser.add_argument("--keywords", nargs="*", default=None)
    args = parser.parse_args()

    run(args.ticker, args.cashtag, args.company, args.days, args.keywords)
