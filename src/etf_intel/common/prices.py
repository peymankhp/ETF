"""Point-in-time price primitives shared by features and labeling.

Lives in ``common`` (the base layer) because both ``features`` and ``labeling``
need the total-return index but, being sibling layers, may not import each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_intel.common.types import Cols


def total_return_index(ticker_bars: pd.DataFrame) -> pd.Series:
    """Compute a point-in-time total-return index for one ticker.

    Built from daily total returns ``(close_t + div_t) / (close_{t-1} / split_t) - 1``
    seeded at the first close, so the value at ``T`` depends only on data available
    at ``T`` (no future-dividend back-adjustment leak — Decision C).

    Args:
        ticker_bars: One ticker's rows, sorted by date, with columns
            ``close``, ``dividends``, ``splits``.

    Returns:
        A float Series (same index as ``ticker_bars``) of the total-return index.
    """
    close = ticker_bars[Cols.CLOSE].astype(float)
    div = ticker_bars[Cols.DIVIDENDS].astype(float).fillna(0.0)
    split = ticker_bars[Cols.SPLITS].astype(float).replace(0.0, 1.0).fillna(1.0)

    prev_close_adj = close.shift(1) / split
    daily_tr = ((close + div) / prev_close_adj - 1.0).fillna(0.0)

    index = close.iloc[0] * (1.0 + daily_tr).cumprod()
    index.iloc[0] = float(close.iloc[0])
    return index.astype(float).replace([np.inf, -np.inf], np.nan)
