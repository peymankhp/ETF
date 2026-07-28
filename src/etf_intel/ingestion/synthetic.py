"""Deterministic synthetic providers for offline runs, CI, and tests.

Given the same seed and inputs these produce byte-identical output, which keeps
the whole pipeline reproducible without network access or API keys.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from etf_intel.common.types import Cols
from etf_intel.ingestion.base import MacroDataProvider, MarketDataProvider


def _ticker_seed(base_seed: int, ticker: str) -> int:
    """Derive a stable per-ticker seed from a base seed and the symbol.

    Uses a deterministic SHA-256 digest (not the salted built-in ``hash``) so
    output is identical across processes and runs.
    """
    digest = hashlib.sha256(f"etf-intel:{ticker}".encode()).hexdigest()
    h = int(digest[:8], 16)
    return (base_seed * 1_000_003 + h) % (2**31)


class SyntheticMarketProvider(MarketDataProvider):
    """Generates reproducible geometric-Brownian-motion price paths."""

    def __init__(self, seed: int = 42, annual_drift: float = 0.06, annual_vol: float = 0.18):
        """Initialise the generator.

        Args:
            seed: Base random seed.
            annual_drift: Mean annualised drift of the price paths.
            annual_vol: Annualised volatility of the price paths.
        """
        self.seed = seed
        self.annual_drift = annual_drift
        self.annual_vol = annual_vol

    def fetch(
        self,
        tickers: list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic unadjusted OHLCV for the given tickers."""
        end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
        dates = pd.bdate_range(pd.Timestamp(start), end_ts)
        n = len(dates)
        dt = 1.0 / 252.0
        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            rng = np.random.default_rng(_ticker_seed(self.seed, ticker))
            shocks = rng.normal(
                (self.annual_drift - 0.5 * self.annual_vol**2) * dt,
                self.annual_vol * np.sqrt(dt),
                size=n,
            )
            start_price = float(rng.uniform(40.0, 400.0))
            close = start_price * np.exp(np.cumsum(shocks))
            intraday = np.abs(rng.normal(0.0, 0.004, size=n)) * close
            open_ = close * (1.0 + rng.normal(0.0, 0.002, size=n))
            high = np.maximum(open_, close) + intraday
            low = np.minimum(open_, close) - intraday
            volume = rng.integers(1_000_000, 30_000_000, size=n)
            frames.append(
                pd.DataFrame(
                    {
                        Cols.DATE: dates,
                        Cols.TICKER: ticker,
                        Cols.OPEN: open_,
                        Cols.HIGH: high,
                        Cols.LOW: low,
                        Cols.CLOSE: close,
                        Cols.VOLUME: volume,
                        Cols.DIVIDENDS: 0.0,
                        Cols.SPLITS: 1.0,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


class SyntheticMacroProvider(MacroDataProvider):
    """Generates reproducible slow-moving macro series."""

    def __init__(self, seed: int = 42):
        """Initialise the generator.

        Args:
            seed: Base random seed.
        """
        self.seed = seed

    def fetch(
        self,
        series_ids: list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic monthly macro observations."""
        end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
        dates = pd.date_range(pd.Timestamp(start), end_ts, freq="MS")
        n = len(dates)
        frames: list[pd.DataFrame] = []
        for sid in series_ids:
            rng = np.random.default_rng(_ticker_seed(self.seed, sid))
            level = float(rng.uniform(1.0, 5.0))
            walk = level + np.cumsum(rng.normal(0.0, 0.05, size=n))
            frames.append(pd.DataFrame({Cols.DATE: dates, "series_id": sid, "value": walk}))
        return pd.concat(frames, ignore_index=True)
