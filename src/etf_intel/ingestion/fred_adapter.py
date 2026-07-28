"""FRED macro-data adapter via ``fredapi``."""

from __future__ import annotations

import pandas as pd

from etf_intel.common.logging import get_logger
from etf_intel.ingestion.base import MacroDataProvider

logger = get_logger(__name__)


class FredMacroProvider(MacroDataProvider):
    """Fetches macro series from FRED (free API key required)."""

    def __init__(self, api_key: str):
        """Initialise with a FRED API key.

        Args:
            api_key: FRED API key from the environment.

        Raises:
            ValueError: If the key is empty.
        """
        if not api_key:
            raise ValueError(
                "FRED API key is required. Set ETF_INTEL_FRED_API_KEY or use "
                "--source synthetic for offline runs."
            )
        self.api_key = api_key

    def fetch(
        self,
        series_ids: list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch each FRED series and return them in long format."""
        from fredapi import Fred

        fred = Fred(api_key=self.api_key)
        frames: list[pd.DataFrame] = []
        for sid in series_ids:
            logger.info("Fetching %s from FRED", sid)
            series = fred.get_series(
                sid,
                observation_start=str(start),
                observation_end=None if end is None else str(end),
            )
            df = series.rename("value").reset_index(names="date")
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df["series_id"] = sid
            frames.append(df[["date", "series_id", "value"]])
        return pd.concat(frames, ignore_index=True)
