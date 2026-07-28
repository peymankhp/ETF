"""yfinance market-data adapter.

Fetches **unadjusted** OHLCV with corporate actions (dividends, splits) so that
price adjustment is recomputed deterministically downstream, preserving
reproducibility (Decision C).
"""

from __future__ import annotations

import pandas as pd

from etf_intel.common.logging import get_logger
from etf_intel.common.types import Cols
from etf_intel.ingestion.base import MarketDataProvider

logger = get_logger(__name__)


class YFinanceMarketProvider(MarketDataProvider):
    """Fetches raw ETF bars via the free ``yfinance`` API."""

    def fetch(
        self,
        tickers: list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch unadjusted OHLCV + actions for each ticker.

        Args:
            tickers: ETF symbols.
            start: Inclusive start date.
            end: Inclusive end date; ``None`` means today.

        Returns:
            Long-format raw market frame (see :class:`MarketDataProvider`).
        """
        import yfinance as yf

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            logger.info("Fetching %s from yfinance", ticker)
            hist = yf.Ticker(ticker).history(
                start=str(start),
                end=None if end is None else str(pd.Timestamp(end) + pd.Timedelta(days=1)),
                interval="1d",
                auto_adjust=False,  # keep raw unadjusted prices
                actions=True,
            )
            if hist.empty:
                logger.warning("No data returned for %s", ticker)
                continue
            hist = hist.reset_index()
            hist[Cols.DATE] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.normalize()
            frames.append(
                pd.DataFrame(
                    {
                        Cols.DATE: hist[Cols.DATE],
                        Cols.TICKER: ticker,
                        Cols.OPEN: hist["Open"].astype(float),
                        Cols.HIGH: hist["High"].astype(float),
                        Cols.LOW: hist["Low"].astype(float),
                        Cols.CLOSE: hist["Close"].astype(float),
                        Cols.VOLUME: hist["Volume"].astype("int64"),
                        Cols.DIVIDENDS: hist.get("Dividends", 0.0).astype(float),
                        Cols.SPLITS: hist.get("Stock Splits", 0.0).replace(0.0, 1.0).astype(float),
                    }
                )
            )
        if not frames:
            raise RuntimeError("yfinance returned no data for any requested ticker.")
        return pd.concat(frames, ignore_index=True)
