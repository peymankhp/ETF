"""Causal technical indicators (hand-rolled — Decision A, no pandas-ta).

Every function uses only trailing / current values (``rolling``, ``ewm`` with
``adjust=False``, ``pct_change``), so a value at index ``t`` never depends on any
row ``> t``. This is what the leakage test verifies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_intel.common.config import FeaturesConfig
from etf_intel.common.dates import TRADING_DAYS_PER_YEAR


def rolling_return(series: pd.Series, window: int) -> pd.Series:
    """Trailing ``window``-day simple return: ``s_t / s_{t-window} - 1``."""
    return series.pct_change(window)


def rolling_volatility(daily_returns: pd.Series, window: int) -> pd.Series:
    """Annualised trailing volatility of daily returns."""
    return daily_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over a trailing window."""
    return series.rolling(window).mean()


def ma_distance(series: pd.Series, window: int) -> pd.Series:
    """Fractional distance of price above/below its ``window``-day SMA."""
    return series / sma(series, window) - 1.0


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI over a trailing window (0-100)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    # avg_loss == 0 (no down moves) -> rs = +inf -> RSI = 100, the standard convention.
    rs = avg_gain / avg_loss
    rsi_values = 100.0 - 100.0 / (1.0 + rs)
    # Flat stretch (no gains and no losses) leaves RSI undefined; mark as NaN.
    rsi_values = rsi_values.mask((avg_gain == 0) & (avg_loss == 0), np.nan)
    return rsi_values


def macd_hist_normalised(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """MACD histogram normalised by price (scale-free, causal).

    Normalising by the current level makes the feature invariant to the index's
    arbitrary starting scale while remaining point-in-time.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - signal_line) / series


def rolling_drawdown(series: pd.Series, window: int) -> pd.Series:
    """Current drawdown from the trailing ``window``-day high (<= 0)."""
    roll_max = series.rolling(window, min_periods=1).max()
    return series / roll_max - 1.0


def compute_technical(index_series: pd.Series, cfg: FeaturesConfig) -> pd.DataFrame:
    """Compute all technical features for one ticker's total-return index.

    Args:
        index_series: Point-in-time total-return index (sorted by date).
        cfg: Feature configuration.

    Returns:
        A frame of technical features aligned to ``index_series``' index.
    """
    daily_ret = index_series.pct_change()
    out: dict[str, pd.Series] = {}

    for w in cfg.return_windows:
        out[f"ret_{w}"] = rolling_return(index_series, w)
    for w in cfg.vol_windows:
        out[f"vol_{w}"] = rolling_volatility(daily_ret, w)
    for w in cfg.ma_windows:
        out[f"ma_dist_{w}"] = ma_distance(index_series, w)

    out[f"rsi_{cfg.rsi_window}"] = rsi(index_series, cfg.rsi_window)
    out["macd_hist"] = macd_hist_normalised(
        index_series, cfg.macd.fast, cfg.macd.slow, cfg.macd.signal
    )
    out[f"drawdown_{cfg.drawdown_window}"] = rolling_drawdown(index_series, cfg.drawdown_window)
    return pd.DataFrame(out)
