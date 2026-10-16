"""Fetch historical OHLCV price bars for the asset, aligned to the chosen bar size."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from config import CFG

_INTERVAL_MAP = {"1h": "60m", "1d": "1d"}


def fetch_price_history(ticker: str, lookback_days: int = CFG.lookback_days, bar_size: str = CFG.bar_size) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by UTC timestamp with columns:
    open, high, low, close, volume.

    Note: yfinance intraday (60m) history is limited to ~730 days, which is
    fine for a 180-day lookback. For daily bars there's no such limit.
    """
    interval = _INTERVAL_MAP.get(bar_size, "60m")
    period = f"{min(lookback_days + 5, 730)}d"

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No price data returned for {ticker}")

    df = df.reset_index()
    ts_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={ts_col: "timestamp"})
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ticker"] = ticker

    keep = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
    return df[keep].sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    bars = fetch_price_history("TSLA")
    print(bars.tail())
    print(f"{len(bars)} bars fetched")
