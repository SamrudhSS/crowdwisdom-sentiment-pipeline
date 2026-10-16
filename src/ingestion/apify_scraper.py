"""
Scrapes X (Twitter) posts for a ticker/cashtag/keyword over the last N days
using an Apify actor.

Reproducibility: the actor id + exact run input are logged to
reports/run_manifest.json by the pipeline orchestrator so the reviewer can
re-run the identical Apify job with their own token.
"""
from __future__ import annotations

import datetime as dt
import time
from typing import Any

import requests

from config import CFG

APIFY_BASE = "https://api.apify.com/v2"


def build_search_queries(ticker: str, cashtag: str | None, keywords: list[str] | None) -> list[str]:
    queries = [ticker]
    if cashtag:
        queries.append(cashtag)
    if keywords:
        queries.extend(keywords)
    return list(dict.fromkeys(queries))  # de-dupe, preserve order


def run_apify_actor(search_terms: list[str], since: dt.date, until: dt.date, max_items: int = 20000,
                     sort: str = "Latest") -> list[dict[str, Any]]:
    """
    Kicks off an Apify actor run synchronously (run-sync-get-dataset-items)
    and returns the raw list of post items.

    Actor: apidojo/tweet-scraper ("Tweet Scraper"). Per its documented input
    schema, date filtering (`start`/`end`) only applies to `searchTerms`
    queries (not startUrls/twitterHandles), which is what we use here.
    Note from the actor's docs: it enforces a 50-tweet minimum per query and
    bills per 1,000 tweets ($0.40/1k) -- broad, high-volume search terms
    work best; very narrow queries may return nothing or fail the minimum.
    If a query is too narrow for this actor, the docs recommend the
    "Twitter Scraper Unlimited" actor instead (no minimum-result floor).
    """
    if not CFG.apify_token:
        raise RuntimeError("APIFY_TOKEN not set. Add it to your .env file.")

    url = f"{APIFY_BASE}/acts/{CFG.apify_x_actor_id.replace('/', '~')}/run-sync-get-dataset-items"
    params = {"token": CFG.apify_token}
    payload = {
        "searchTerms": search_terms,
        "start": since.isoformat(),
        "end": until.isoformat(),
        "maxItems": max_items,
        "sort": sort,  # "Latest", "Top", or "Latest + Top"
        "tweetLanguage": "en",
    }

    resp = requests.post(url, params=params, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def normalize_posts(raw_items: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    """
    Map raw apidojo/tweet-scraper output items to a flat, DB-ready schema.

    Matches the actor's actual documented output fields:
      id, url, text, retweetCount, replyCount, likeCount, quoteCount,
      createdAt (e.g. "Fri Nov 24 17:49:36 +0000 2023"), bookmarkCount,
      author.userName, author.followers.
    There is no `viewCount` field in this actor's output, so views are
    left at 0 (kept in the schema for forward-compatibility / other actors).
    """
    rows = []
    for item in raw_items:
        if item.get("type") not in (None, "tweet"):
            continue  # skip non-tweet items (e.g. cursors) if ever present
        try:
            author = item.get("author") or {}
            rows.append(
                {
                    "ticker": ticker,
                    "post_id": str(item.get("id", "")),
                    "created_at_utc": item.get("createdAt", ""),
                    "text": item.get("text", ""),
                    "likes": int(item.get("likeCount", 0) or 0),
                    "reposts": int(item.get("retweetCount", 0) or 0),
                    "replies": int(item.get("replyCount", 0) or 0),
                    "views": int(item.get("viewCount", 0) or 0),
                    "author_handle": author.get("userName", ""),
                    "author_followers": int(author.get("followers", 0) or 0),
                    "url": item.get("url", ""),
                }
            )
        except Exception:
            continue
    return rows


def scrape_x_posts(ticker: str, cashtag: str | None = None, keywords: list[str] | None = None,
                    lookback_days: int = CFG.lookback_days) -> list[dict[str, Any]]:
    until = dt.date.today()
    since = until - dt.timedelta(days=lookback_days)
    search_terms = build_search_queries(ticker, cashtag, keywords)

    raw = run_apify_actor(search_terms, since, until)
    return normalize_posts(raw, ticker)


if __name__ == "__main__":
    posts = scrape_x_posts("TSLA", cashtag="$TSLA", keywords=["Tesla stock"])
    print(f"Scraped {len(posts)} posts")
