"""Abstract provider interfaces for market and macro data.

Concrete adapters (yfinance, FRED, synthetic) implement these so a paid provider
can be swapped in later without touching downstream code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    """Interface for a source of raw, unadjusted ETF OHLCV + corporate actions."""

    @abstractmethod
    def fetch(
        self,
        tickers: list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch raw daily bars.

        Args:
            tickers: ETF symbols to fetch.
            start: Inclusive start date.
            end: Inclusive end date; ``None`` means "up to today".

        Returns:
            Long-format frame with columns
            ``[date, ticker, open, high, low, close, volume, dividends, splits]``.
            Prices are **unadjusted**; ``splits`` is the split ratio on the day
            (1.0 if none) and ``dividends`` the cash amount (0.0 if none).
        """
        raise NotImplementedError


class MacroDataProvider(ABC):
    """Interface for a source of macroeconomic time series (e.g. FRED)."""

    @abstractmethod
    def fetch(
        self,
        series_ids: list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch raw macro series.

        Args:
            series_ids: Provider series identifiers (e.g. FRED codes).
            start: Inclusive start date.
            end: Inclusive end date; ``None`` means "up to today".

        Returns:
            Long-format frame with columns ``[date, series_id, value]``. Dates are
            the series' native observation dates (publication lag handled later in
            ``features`` per :class:`~etf_intel.common.config.FeaturesConfig`).
        """
        raise NotImplementedError
