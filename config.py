"""Central configuration loaded from environment / .env."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    apify_token: str = os.getenv("APIFY_TOKEN", "")
    apify_x_actor_id: str = os.getenv("APIFY_X_ACTOR_ID", "apidojo/tweet-scraper")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    # Tavily credit cost: "basic" = 1 credit/query, "advanced" = 2 credits/query.
    # Free "Researcher" plan = 1,000 credits/month, so default to basic to
    # conserve credits during development; bump to "advanced" for the final
    # submission run if you want higher-quality ranked results.
    tavily_search_depth: str = os.getenv("TAVILY_SEARCH_DEPTH", "basic")
    exa_api_key: str = os.getenv("EXA_API_KEY", "")
    price_provider: str = os.getenv("PRICE_PROVIDER", "yfinance")

    db_path: str = os.getenv("DB_PATH", "data/warehouse.duckdb")
    reports_dir: str = os.getenv("REPORTS_DIR", "reports")
    charts_dir: str = os.getenv("CHARTS_DIR", "charts")
    logs_dir: str = os.getenv("LOGS_DIR", "logs")

    # bar size for price/sentiment alignment: "1h" or "1d"
    bar_size: str = os.getenv("BAR_SIZE", "1h")
    lookback_days: int = int(os.getenv("LOOKBACK_DAYS", "180"))
    horizons: tuple = (0, 1, 2, 3, 4, 5)


CFG = Config()
