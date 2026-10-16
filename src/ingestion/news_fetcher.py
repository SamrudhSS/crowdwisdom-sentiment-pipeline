"""Fetch recent/relevant news for the asset via Tavily (with Exa as optional fallback)."""
from __future__ import annotations

from typing import Any

import requests

from config import CFG

TAVILY_URL = "https://api.tavily.com/search"
EXA_URL = "https://api.exa.ai/search"


def fetch_tavily_news(query: str, max_results: int = 15, days: int = 7,
                       search_depth: str | None = None) -> list[dict[str, Any]]:
    """
    search_depth: "basic" (1 credit/query) or "advanced" (2 credits/query).
    Defaults to CFG.tavily_search_depth (itself defaulting to "basic") to
    conserve API credits; pass search_depth="advanced" explicitly for a
    higher-quality final run.
    """
    if not CFG.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY not set. Add it to your .env file.")

    payload = {
        "api_key": CFG.tavily_api_key,
        "query": query,
        "topic": "news",
        "search_depth": search_depth or CFG.tavily_search_depth,
        "max_results": max_results,
        "days": days,
        "include_answer": False,
    }
    resp = requests.post(TAVILY_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "published": r.get("published_date", ""),
            "score": r.get("score", 0.0),
            "source": "tavily",
        }
        for r in data.get("results", [])
    ]


def fetch_exa_news(query: str, max_results: int = 15) -> list[dict[str, Any]]:
    """Optional secondary source, included for triangulation / redundancy."""
    if not CFG.exa_api_key:
        return []
    headers = {"x-api-key": CFG.exa_api_key}
    payload = {"query": query, "numResults": max_results, "type": "neural", "category": "news"}
    resp = requests.post(EXA_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("text", ""),
            "published": r.get("publishedDate", ""),
            "score": r.get("score", 0.0),
            "source": "exa",
        }
        for r in data.get("results", [])
    ]


def fetch_asset_news(ticker: str, company_name: str | None = None) -> list[dict[str, Any]]:
    query = f"{company_name or ticker} ({ticker}) stock news"
    news = fetch_tavily_news(query)
    news += fetch_exa_news(query)
    return news


if __name__ == "__main__":
    items = fetch_asset_news("TSLA", "Tesla")
    for n in items[:5]:
        print(n["title"], "-", n["url"])
