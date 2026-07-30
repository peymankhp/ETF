"""Point-in-time-safe fundamental features derived from owned data.

Free historical fundamentals (expense ratio, AUM, current yield from yfinance
``.info``) are only available as *today's* value, so using them across history
would leak the future. Instead we derive the one fundamental we can reconstruct
point-in-time from data we already own: the trailing dividend yield (past
dividends over the current price). Expense-ratio / AUM factors are deferred until
a vintaged fundamentals source is available.
"""

from __future__ import annotations

import pandas as pd

from etf_intel.common.types import Cols


def trailing_dividend_yield(ticker_bars: pd.DataFrame, window: int = 252) -> pd.Series:
    """Trailing dividend yield: sum of dividends over ``window`` days / current price.

    Uses only past dividends and the current close, so it is point-in-time correct.

    Args:
        ticker_bars: One ticker's rows, sorted by date, with ``dividends`` and
            ``close`` columns.
        window: Trailing window in trading days (≈252 = 1 year).

    Returns:
        A float Series (same index) of the trailing dividend yield.
    """
    div_ttm = ticker_bars[Cols.DIVIDENDS].fillna(0.0).rolling(window, min_periods=window // 4).sum()
    return (div_ttm / ticker_bars[Cols.CLOSE]).astype(float)
