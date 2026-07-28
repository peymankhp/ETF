"""Release-lagged macro features (Decision B).

Each FRED series is shifted forward by a conservative publication lag so a feature
at date ``T`` only uses a value the market had actually seen by ``T``. Values are
then carried forward onto trading dates via a backward as-of join.
"""

from __future__ import annotations

import pandas as pd

from etf_intel.common.config import FeaturesConfig
from etf_intel.common.types import Cols

DEFAULT_LAG_DAYS = 30
MACRO_CHANGE_WINDOW = 63  # ~3 months of trading days


def build_macro_features(
    macro_df: pd.DataFrame,
    trading_dates: pd.Series,
    cfg: FeaturesConfig,
    series_map: dict[str, str],
) -> pd.DataFrame:
    """Build daily, release-lagged macro features aligned to trading dates.

    Args:
        macro_df: Long macro frame ``[date, series_id, value]``.
        trading_dates: Sorted unique trading dates to align onto.
        cfg: Feature configuration (holds per-series release lags).
        series_map: FRED id -> friendly name.

    Returns:
        A frame with a ``date`` column plus ``macro_<name>`` level and
        ``macro_<name>_chg63`` change columns.
    """
    base = pd.DataFrame({Cols.DATE: pd.to_datetime(sorted(trading_dates.unique()))})
    if macro_df is None or macro_df.empty:
        return base

    for sid, name in series_map.items():
        sdf = macro_df[macro_df["series_id"] == sid].copy()
        if sdf.empty:
            continue
        lag = cfg.macro_release_lag_days.get(sid, DEFAULT_LAG_DAYS)
        sdf[Cols.DATE] = pd.to_datetime(sdf[Cols.DATE]) + pd.Timedelta(days=lag)
        sdf = sdf.sort_values(Cols.DATE)[[Cols.DATE, "value"]]
        merged = pd.merge_asof(base, sdf, on=Cols.DATE, direction="backward")
        base[f"macro_{name}"] = merged["value"].to_numpy()
        base[f"macro_{name}_chg{MACRO_CHANGE_WINDOW}"] = (
            merged["value"].diff(MACRO_CHANGE_WINDOW).to_numpy()
        )
    return base
