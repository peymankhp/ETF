"""Typed domain records, rating enum, and canonical column names.

Keeping column names in one place avoids stringly-typed drift across modules and
gives a single source of truth for the on-disk / in-memory schema.
"""

from __future__ import annotations

from enum import StrEnum


class Rating(StrEnum):
    """The six ordered rating buckets, best to worst."""

    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    REDUCE = "Reduce"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"


def rating_order() -> list[Rating]:
    """Return ratings ordered from best (Strong Buy) to worst (Strong Sell)."""
    return [
        Rating.STRONG_BUY,
        Rating.BUY,
        Rating.HOLD,
        Rating.REDUCE,
        Rating.SELL,
        Rating.STRONG_SELL,
    ]


class Cols:
    """Canonical column names used across frames (single source of truth)."""

    # Identity
    DATE = "date"
    TICKER = "ticker"

    # Raw OHLCV snapshot (unadjusted) + corporate actions
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    DIVIDENDS = "dividends"
    SPLITS = "splits"

    # Deterministically derived
    ADJ_CLOSE = "adj_close"

    # Model / ranking outputs
    SCORE = "score"
    PROB_OUTPERFORM = "prob_outperform"
    RANK = "rank"
    RATING = "rating"


# Raw market snapshot schema (long format, one row per ticker-date).
MARKET_SCHEMA: tuple[str, ...] = (
    Cols.DATE,
    Cols.TICKER,
    Cols.OPEN,
    Cols.HIGH,
    Cols.LOW,
    Cols.CLOSE,
    Cols.VOLUME,
    Cols.DIVIDENDS,
    Cols.SPLITS,
)
