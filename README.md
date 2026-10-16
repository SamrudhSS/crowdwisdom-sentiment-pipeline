# CrowdWisdom X Sentiment Price Prediction Agent

A Python pipeline that scrapes X (Twitter) posts and news for a given asset,
joins them with price history in a local database, engineers sentiment +
price features, and trains a walk-forward-validated multi-horizon model to
predict price at **n+0, n+1, n+2, n+3, n+4, n+5** bars ahead.

## Architecture

```
X (Apify) ──┐
             ├─► SQLite/DuckDB ──► Feature Engineering ──► Walk-Forward ML ──► Eval + Charts
News (Tavily)┘        ▲
Price (yfinance) ──────┘
```

## Repo layout

```
config.py                      # env/config loader
src/ingestion/apify_scraper.py # X post scraping via Apify actor
src/ingestion/news_fetcher.py  # Tavily news ingestion
src/ingestion/price_fetcher.py # historical OHLCV bars
src/db/database.py             # schema + join logic (DuckDB)
src/features/sentiment_scoring.py  # VADER + engagement-weighted sentiment
src/features/feature_engineering.py# lagged returns, rolling sentiment, horizons
src/models/train.py            # walk-forward, multi-horizon LightGBM
src/models/evaluate.py         # Sharpe / Sortino / Max Drawdown
src/viz/charts.py              # sentiment-vs-price matrix charts
src/pipeline.py                # orchestrator (run end-to-end)
reports/evaluation_report_template.md
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in APIFY_TOKEN, TAVILY_API_KEY
```

## Run

```bash
python -m src.pipeline --ticker TSLA --cashtag "$TSLA" --days 180
```

This will:
1. Scrape ~180 days of X posts mentioning the ticker/cashtag via Apify.
2. Pull 180 days of OHLCV bars (yfinance, bar size configurable).
3. Pull recent news via Tavily for the 1-page sentiment summary.
4. Join everything into `data/warehouse.duckdb`.
5. Engineer features, run walk-forward training for 6 horizons (n+0..n+5).
6. Write `reports/evaluation_report.md` with Sharpe/Sortino/MaxDD per horizon.
7. Write `charts/sentiment_price_matrix.png` and related plots.

## Apify / Tavily reproducibility

See `CREDENTIALS.md` (not committed — fill in locally and send to the
reviewer separately, never commit real keys). The actor ID used for X
scraping and the exact Tavily search parameters are logged in
`src/ingestion/apify_scraper.py` and `src/ingestion/news_fetcher.py`
as named constants, and are also echoed into `reports/run_manifest.json`
on every pipeline run for exact reproducibility.

## Notes / assumptions

- "Price only" + "sentiment only" features: no other alternative data
  (options flow, on-chain, etc.) is used in the feature set by design.
- Bar granularity defaults to 1-hour bars for an 180-day window (configurable
  to daily); n+1..n+5 are expressed in bars, not calendar days.
- Walk-forward validation uses expanding-window train / fixed-size
  out-of-sample test folds, strictly time-ordered (no shuffling, no leakage).
