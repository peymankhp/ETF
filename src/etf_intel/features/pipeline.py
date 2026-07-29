"""Assemble the full point-in-time feature matrix from raw snapshots.

Orchestrates: total-return index -> per-ticker technical indicators ->
cross-sectional (relative strength + same-date ranks) -> release-lagged macro.
Output is a long frame keyed by ``(date, ticker)``.
"""

from __future__ import annotations

import pandas as pd

from etf_intel.common.config import AppConfig, Universe
from etf_intel.common.prices import total_return_index
from etf_intel.common.types import Cols
from etf_intel.features.cross_sectional import cross_sectional_rank, relative_strength
from etf_intel.features.fundamentals import trailing_dividend_yield
from etf_intel.features.macro import build_macro_features
from etf_intel.features.technical import compute_technical, rolling_return

# Columns that identify a row or are non-stationary references — never model inputs.
NON_FEATURE_COLS: frozenset[str] = frozenset({Cols.DATE, Cols.TICKER, Cols.ADJ_CLOSE})


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Return the model-input feature columns (excludes id / reference columns)."""
    return [c for c in features.columns if c not in NON_FEATURE_COLS]


def build_features(
    market_df: pd.DataFrame,
    macro_df: pd.DataFrame | None,
    universe: Universe,
    config: AppConfig,
) -> pd.DataFrame:
    """Build the point-in-time feature matrix.

    Args:
        market_df: Raw long market snapshot.
        macro_df: Raw long macro snapshot (may be ``None``/empty).
        universe: ETF universe (provides the benchmark).
        config: Application config.

    Returns:
        Long feature frame ``[date, ticker, adj_close, <features...>]`` sorted by
        ``(date, ticker)``, with warm-up rows lacking a 21-day return dropped.
    """
    fcfg = config.features
    market = market_df.sort_values([Cols.TICKER, Cols.DATE]).reset_index(drop=True)

    per_ticker: list[pd.DataFrame] = []
    for ticker, bars in market.groupby(Cols.TICKER, sort=False):
        bars = bars.reset_index(drop=True)
        idx = total_return_index(bars)
        tech = compute_technical(idx, fcfg)
        # Point-in-time fundamental: trailing dividend yield (past divs / price).
        tech["ttm_yield"] = trailing_dividend_yield(bars, fcfg.yield_window).to_numpy()
        tech.insert(0, Cols.ADJ_CLOSE, idx.to_numpy())
        tech.insert(0, Cols.TICKER, ticker)
        tech.insert(0, Cols.DATE, bars[Cols.DATE].to_numpy())
        per_ticker.append(tech)

    feat = pd.concat(per_ticker, ignore_index=True)

    # Cross-sectional: relative strength vs benchmark + same-date momentum rank.
    rs_window = fcfg.rel_strength_window
    bench = feat[feat[Cols.TICKER] == universe.benchmark].sort_values(Cols.DATE)
    bench_ret = (
        bench.set_index(Cols.DATE)[Cols.ADJ_CLOSE]
        .pipe(rolling_return, rs_window)
        .rename("bench_ret")
    )
    ret_col = f"ret_{rs_window}"
    if ret_col in feat.columns:
        feat["rel_strength"] = relative_strength(feat, bench_ret, rs_window)
    if "ret_21" in feat.columns:
        feat["xs_mom_rank"] = cross_sectional_rank(feat, "ret_21")

    # Macro (release-lagged), broadcast across tickers by date. Macro is constant
    # across tickers on a given date, so it adds nothing to cross-sectional ranking;
    # include it only when explicitly enabled (e.g. for a future market-timing model).
    if fcfg.include_macro:
        macro_feat = build_macro_features(
            macro_df if macro_df is not None else pd.DataFrame(),
            feat[Cols.DATE],
            fcfg,
            config.macro_series,
        )
        if macro_feat.shape[1] > 1:  # more than just the date column
            feat = feat.merge(macro_feat, on=Cols.DATE, how="left")

    feat = feat.sort_values([Cols.DATE, Cols.TICKER]).reset_index(drop=True)
    if "ret_21" in feat.columns:
        feat = feat.dropna(subset=["ret_21"]).reset_index(drop=True)
    return feat
