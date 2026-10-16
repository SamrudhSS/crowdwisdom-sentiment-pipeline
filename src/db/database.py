"""
DuckDB-backed warehouse. Holds raw posts, raw news, raw price bars, and a
joined bar-level table (price + aggregated sentiment per bar).

DuckDB is used instead of SQLite because it is columnar, embedded (no
server), and fast for the time-series aggregation/window-function joins
used here -- while still being a single portable file like SQLite.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from config import CFG

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS x_posts (
    ticker            VARCHAR,
    post_id           VARCHAR,
    created_at_utc    TIMESTAMP,
    text              VARCHAR,
    likes             INTEGER,
    reposts           INTEGER,
    replies           INTEGER,
    views             BIGINT,
    author_handle     VARCHAR,
    author_followers  BIGINT,
    url               VARCHAR,
    sentiment_compound DOUBLE,
    PRIMARY KEY (ticker, post_id)
);

CREATE TABLE IF NOT EXISTS news (
    ticker      VARCHAR,
    title       VARCHAR,
    url         VARCHAR,
    content     VARCHAR,
    published   VARCHAR,
    score       DOUBLE,
    source      VARCHAR
);

CREATE TABLE IF NOT EXISTS price_bars (
    ticker     VARCHAR,
    timestamp  TIMESTAMP,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    volume     BIGINT,
    PRIMARY KEY (ticker, timestamp)
);

CREATE TABLE IF NOT EXISTS bar_features (
    ticker               VARCHAR,
    timestamp            TIMESTAMP,
    close                DOUBLE,
    ret_1                DOUBLE,
    post_count           INTEGER,
    sentiment_mean       DOUBLE,
    sentiment_engagement_wtd DOUBLE,
    follower_wtd_sentiment DOUBLE,
    PRIMARY KEY (ticker, timestamp)
);
"""


def get_connection(db_path: str = CFG.db_path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path)
    con.execute(SCHEMA_SQL)
    return con

def upsert_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, keys: list[str]) -> None:
    if df.empty:
        return
    con.register("tmp_df", df)
    cols = ", ".join(df.columns)
    key_clause = " AND ".join([f"{table}.{k} = tmp_df.{k}" for k in keys])
    con.execute(f"DELETE FROM {table} USING tmp_df WHERE {key_clause}")
    con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM tmp_df")
    con.unregister("tmp_df")

def save_posts(con, df: pd.DataFrame) -> None:
    upsert_df(con, "x_posts", df, ["ticker", "post_id"])


def save_news(con, df: pd.DataFrame) -> None:
    if df.empty:
        return
    con.register("tmp_news", df)
    cols = ", ".join(df.columns)
    con.execute(f"INSERT INTO news ({cols}) SELECT {cols} FROM tmp_news")
    con.unregister("tmp_news")


def save_price_bars(con, df: pd.DataFrame) -> None:
    upsert_df(con, "price_bars", df, ["ticker", "timestamp"])


def build_bar_features(con, ticker: str, bar_size: str = CFG.bar_size) -> pd.DataFrame:
    """
    Aggregates posts into the same bar grid as price (floor timestamp to the
    bar size), computing count/mean/engagement-weighted/follower-weighted
    sentiment per bar, then joins onto price bars with a 1-bar-forward-fill
    for sentiment (posts between bars attach to the bar they fall in).
    """
    freq = "hour" if bar_size == "1h" else "day"

    query = f"""
    WITH posts_binned AS (
        SELECT
            ticker,
            date_trunc('{freq}', created_at_utc) AS timestamp,
            sentiment_compound,
            (1 + likes + 2*reposts + 0.5*replies) AS engagement_weight,
            GREATEST(author_followers, 1) AS followers
        FROM x_posts
        WHERE ticker = '{ticker}'
    ),
    agg AS (
        SELECT
            ticker,
            timestamp,
            COUNT(*) AS post_count,
            AVG(sentiment_compound) AS sentiment_mean,
            SUM(sentiment_compound * engagement_weight) / NULLIF(SUM(engagement_weight), 0) AS sentiment_engagement_wtd,
            SUM(sentiment_compound * LN(followers)) / NULLIF(SUM(LN(followers)), 0) AS follower_wtd_sentiment
        FROM posts_binned
        GROUP BY ticker, timestamp
    )
    SELECT
        p.ticker,
        p.timestamp,
        p.close,
        p.close / LAG(p.close) OVER (PARTITION BY p.ticker ORDER BY p.timestamp) - 1 AS ret_1,
        COALESCE(a.post_count, 0) AS post_count,
        a.sentiment_mean,
        a.sentiment_engagement_wtd,
        a.follower_wtd_sentiment
    FROM price_bars p
    LEFT JOIN agg a USING (ticker, timestamp)
    WHERE p.ticker = '{ticker}'
    ORDER BY p.timestamp
    """
    df = con.execute(query).fetchdf()
    upsert_df(con, "bar_features", df, ["ticker", "timestamp"])
    return df
