"""
Scores each X post's text for sentiment using VADER (rule-based, fast,
no external API calls -- appropriate given this must run purely on
sentiment + price with no extra paid NLP dependency).

Swap `score_text` for a transformer-based model (e.g. cardiffnlp/twitter-
roberta-base-sentiment) if higher accuracy is needed and latency budget
allows; the rest of the pipeline is agnostic to the scorer.
"""
from __future__ import annotations

import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    text = _URL_RE.sub("", text or "")
    text = _MENTION_RE.sub("", text)
    return text.strip()


def score_text(text: str) -> float:
    """Returns VADER compound score in [-1, 1]."""
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0
    return _analyzer.polarity_scores(cleaned)["compound"]


def score_posts(posts_df: pd.DataFrame) -> pd.DataFrame:
    df = posts_df.copy()
    df["sentiment_compound"] = df["text"].apply(score_text)
    return df
