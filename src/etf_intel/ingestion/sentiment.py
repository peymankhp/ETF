"""News-sentiment scaffold: a swappable scorer + a yfinance headline provider.

IMPORTANT (honest limitation): free historical news does not exist for a 16-year
backtest, so this sentiment signal **cannot be backtested** — it is a *live* overlay
on the current ranking only, deliberately kept out of the backtested model.

The scorer is behind an interface so a heavier model (FinBERT via transformers)
can replace the lightweight lexicon later without touching callers.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import pandas as pd

_WORD = re.compile(r"[a-z']+")

# Tiny finance sentiment lexicon (placeholder for FinBERT).
POSITIVE_WORDS: frozenset[str] = frozenset(
    {
        "beat",
        "beats",
        "surge",
        "surges",
        "surged",
        "upgrade",
        "upgraded",
        "gain",
        "gains",
        "rally",
        "rallies",
        "growth",
        "strong",
        "outperform",
        "record",
        "bullish",
        "jump",
        "jumps",
        "soar",
        "soars",
        "rise",
        "rises",
        "boost",
        "high",
        "profit",
        "profits",
        "win",
        "wins",
        "top",
        "tops",
        "optimistic",
        "recovery",
    }
)
NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "miss",
        "misses",
        "plunge",
        "plunges",
        "plunged",
        "downgrade",
        "downgraded",
        "loss",
        "losses",
        "cut",
        "cuts",
        "weak",
        "fall",
        "falls",
        "fell",
        "decline",
        "declines",
        "bearish",
        "lawsuit",
        "warn",
        "warns",
        "warning",
        "slump",
        "slumps",
        "drop",
        "drops",
        "fear",
        "fears",
        "risk",
        "risks",
        "crash",
        "selloff",
        "recession",
    }
)


class SentimentScorer(ABC):
    """Scores a single text into a sentiment value in ``[-1, 1]``."""

    @abstractmethod
    def score(self, text: str) -> float:
        """Return a sentiment score in ``[-1, 1]`` for ``text``."""
        raise NotImplementedError


class LexiconSentimentScorer(SentimentScorer):
    """Lightweight lexicon scorer (no ML deps); placeholder for FinBERT."""

    def score(self, text: str) -> float:
        """Score as ``(pos - neg) / (pos + neg)`` over matched lexicon words."""
        tokens = _WORD.findall(text.lower())
        pos = sum(t in POSITIVE_WORDS for t in tokens)
        neg = sum(t in NEGATIVE_WORDS for t in tokens)
        if pos + neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)


class FinBERTSentimentScorer(SentimentScorer):
    """FinBERT sentiment scorer (drop-in upgrade from the lexicon).

    Requires the heavy ML stack, which is an optional extra (kept out of the core
    install and CI)::

        uv sync --extra finbert

    The model is loaded lazily on first construction so importing this module stays
    cheap. Returns +score for positive, -score for negative, 0 for neutral.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """Load the FinBERT sentiment pipeline (dynamic transformers import)."""
        import importlib

        # Dynamic import: transformers is optional and heavily typed; loading it via
        # importlib keeps this module type-checkable whether or not it is installed.
        transformers = importlib.import_module("transformers")
        self._pipe: Any = transformers.pipeline(
            "sentiment-analysis", model=model_name, truncation=True
        )

    def score(self, text: str) -> float:
        """Score one text with FinBERT, mapping label+confidence to ``[-1, 1]``."""
        if not text.strip():
            return 0.0
        result = self._pipe(text[:512])[0]
        label = str(result["label"]).lower()
        confidence = float(result["score"])
        if label == "positive":
            return confidence
        if label == "negative":
            return -confidence
        return 0.0


NewsFetcher = Callable[[str], list[dict[str, Any]]]


def _headline(item: dict[str, Any]) -> str:
    """Extract a headline from a yfinance news item (handles old & new schemas)."""
    if item.get("title"):
        return str(item["title"])
    content = item.get("content") or {}
    return str(content.get("title", ""))


def _yfinance_news(ticker: str) -> list[dict[str, Any]]:
    """Fetch recent news items for a ticker via yfinance."""
    import yfinance as yf

    return list(yf.Ticker(ticker).news or [])


class NewsSentimentProvider:
    """Aggregates current-headline sentiment per ticker (live overlay only)."""

    def __init__(
        self, scorer: SentimentScorer | None = None, news_fetcher: NewsFetcher | None = None
    ):
        """Initialise with a scorer and a news fetcher (both injectable for tests)."""
        self.scorer = scorer or LexiconSentimentScorer()
        self.fetch_news = news_fetcher or _yfinance_news

    def fetch(self, tickers: list[str]) -> pd.DataFrame:
        """Return per-ticker mean headline sentiment.

        Args:
            tickers: Symbols to score.

        Returns:
            Frame ``[ticker, sentiment, n_headlines]`` (sentiment NaN if no news).
        """
        rows: list[dict[str, Any]] = []
        for ticker in tickers:
            headlines = [h for h in (_headline(i) for i in self.fetch_news(ticker)) if h]
            scores = [self.scorer.score(h) for h in headlines]
            rows.append(
                {
                    "ticker": ticker,
                    "sentiment": sum(scores) / len(scores) if scores else float("nan"),
                    "n_headlines": len(headlines),
                }
            )
        return pd.DataFrame(rows)
