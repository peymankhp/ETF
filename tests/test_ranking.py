"""Tests for cross-sectional rating assignment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_intel.common.config import RatingsConfig
from etf_intel.common.types import Cols, Rating
from etf_intel.portfolio.ranking import _bucket_counts, assign_ratings


def test_bucket_counts_sum_to_n() -> None:
    fractions = [0.10, 0.20, 0.40, 0.15, 0.10, 0.05]
    for n in (1, 5, 7, 20, 33, 101):
        counts = _bucket_counts(n, fractions)
        assert sum(counts) == n
        assert all(c >= 0 for c in counts)


def test_assign_ratings_best_score_is_strong_buy() -> None:
    dates = pd.to_datetime(["2020-01-31"] * 20)
    scored = pd.DataFrame(
        {
            Cols.DATE: dates,
            Cols.TICKER: [f"T{i}" for i in range(20)],
            Cols.SCORE: np.linspace(0, 1, 20),
        }
    )
    out = assign_ratings(scored, RatingsConfig())
    best = out.sort_values(Cols.SCORE, ascending=False).iloc[0]
    worst = out.sort_values(Cols.SCORE, ascending=False).iloc[-1]
    assert best[Cols.RATING] == Rating.STRONG_BUY.value
    assert worst[Cols.RATING] == Rating.STRONG_SELL.value
    # Ranks are unique 1..20 within the date.
    assert sorted(out[Cols.RANK]) == list(range(1, 21))


def test_assign_ratings_is_per_date() -> None:
    frames = []
    for day in ("2020-01-31", "2020-02-28"):
        frames.append(
            pd.DataFrame(
                {
                    Cols.DATE: pd.to_datetime([day] * 6),
                    Cols.TICKER: list("ABCDEF"),
                    Cols.SCORE: [1, 2, 3, 4, 5, 6],
                }
            )
        )
    out = assign_ratings(pd.concat(frames, ignore_index=True), RatingsConfig())
    # Each date independently ranks 1..6.
    for _, g in out.groupby(Cols.DATE):
        assert sorted(g[Cols.RANK]) == [1, 2, 3, 4, 5, 6]
