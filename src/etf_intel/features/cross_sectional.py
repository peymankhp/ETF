"""Cross-sectional features: relative strength vs benchmark and same-date ranks.

Cross-sectional operations use only values observed on the *same* date across
tickers, so they remain point-in-time correct.
"""

from __future__ import annotations

import pandas as pd

from etf_intel.common.types import Cols


def relative_strength(
    features: pd.DataFrame, benchmark_returns: pd.Series, window: int
) -> pd.Series:
    """Relative strength of each ticker's trailing return vs the benchmark.

    Args:
        features: Long features frame containing ``date`` and ``ret_{window}``.
        benchmark_returns: Benchmark trailing ``window`` returns indexed by date.
        window: The return window (must match a computed ``ret_{window}`` column).

    Returns:
        A Series aligned to ``features`` of ``(1+r_ticker)/(1+r_bench) - 1``.
    """
    col = f"ret_{window}"
    bench = features[Cols.DATE].map(benchmark_returns)
    return (1.0 + features[col]) / (1.0 + bench) - 1.0


def cross_sectional_rank(features: pd.DataFrame, column: str) -> pd.Series:
    """Percentile rank (0-1) of ``column`` within each date across tickers.

    Args:
        features: Long features frame containing ``date`` and ``column``.
        column: Column to rank.

    Returns:
        A Series of per-date percentile ranks (higher = stronger).
    """
    return features.groupby(Cols.DATE)[column].rank(pct=True)
