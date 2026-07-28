"""Performance metrics and equity-curve construction from backtest predictions.

The MVP strategy is a long-only, equal-weight book of the top rating bucket,
rebalanced monthly. Returns are chained from each rebalance date's realised
forward return of the held names, with the benchmark tracked alongside.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from etf_intel.backtest.walkforward import REALIZED_FWD
from etf_intel.common.config import AppConfig, Universe
from etf_intel.common.types import Cols, Rating

PERIODS_PER_YEAR = 12  # monthly rebalance


def perf_stats(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> dict[str, float]:
    """Compute core performance statistics from a periodic return series.

    Args:
        returns: Periodic (monthly) simple returns.
        periods_per_year: Compounding periods per year.

    Returns:
        Dict with CAGR, annualised vol, Sharpe, Sortino, max drawdown, hit rate.
    """
    r = returns.dropna().to_numpy(dtype=float)
    if r.size == 0:
        return {k: float("nan") for k in ("cagr", "vol", "sharpe", "sortino", "max_dd", "hit_rate")}

    equity = np.cumprod(1.0 + r)
    years = r.size / periods_per_year
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")

    ann_ret = float(np.mean(r) * periods_per_year)
    vol = float(np.std(r, ddof=1) * np.sqrt(periods_per_year)) if r.size > 1 else float("nan")
    sharpe = ann_ret / vol if vol and not np.isnan(vol) else float("nan")

    downside = r[r < 0]
    dd_vol = (
        float(np.std(downside, ddof=1) * np.sqrt(periods_per_year))
        if downside.size > 1
        else float("nan")
    )
    sortino = ann_ret / dd_vol if dd_vol and not np.isnan(dd_vol) else float("nan")

    running_max = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / running_max - 1.0))
    hit_rate = float(np.mean(r > 0))

    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "hit_rate": hit_rate,
    }


def compute_backtest(
    predictions: pd.DataFrame, universe: Universe, config: AppConfig
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build the equity curve and metrics for the top-bucket strategy.

    Args:
        predictions: Scored predictions from :func:`run_walk_forward`.
        universe: ETF universe (benchmark).
        config: Application config.

    Returns:
        A ``(metrics, equity_df)`` tuple. ``equity_df`` has columns
        ``date, strategy_return, benchmark_return, strategy_equity,
        benchmark_equity``.
    """
    top_rating = Rating.STRONG_BUY.value
    rows: list[tuple[pd.Timestamp, float, float]] = []

    for d, day in predictions.groupby(Cols.DATE):
        selected = day[day[Cols.RATING] == top_rating]
        if selected.empty:
            # Degenerate top bucket (e.g. tiny universe where 10% rounds to zero
            # names): fall back to the single best-ranked name so the book is held.
            selected = day[day[Cols.RANK] == day[Cols.RANK].min()]
        strat_ret = selected[REALIZED_FWD].mean()
        bench_series = day.loc[day[Cols.TICKER] == universe.benchmark, REALIZED_FWD]
        bench_ret = bench_series.iloc[0] if len(bench_series) else np.nan
        if pd.isna(strat_ret) or pd.isna(bench_ret):
            continue
        rows.append((pd.Timestamp(d), float(strat_ret), float(bench_ret)))

    equity = pd.DataFrame(rows, columns=[Cols.DATE, "strategy_return", "benchmark_return"])
    if not equity.empty:
        equity["strategy_equity"] = (1.0 + equity["strategy_return"]).cumprod()
        equity["benchmark_equity"] = (1.0 + equity["benchmark_return"]).cumprod()

    strat = perf_stats(equity["strategy_return"]) if not equity.empty else perf_stats(pd.Series([]))
    bench = (
        perf_stats(equity["benchmark_return"]) if not equity.empty else perf_stats(pd.Series([]))
    )
    excess = (
        equity["strategy_return"] - equity["benchmark_return"]
        if not equity.empty
        else pd.Series([], dtype=float)
    )
    metrics: dict[str, Any] = {
        "n_periods": int(len(equity)),
        "strategy": strat,
        "benchmark": bench,
        "excess_hit_rate": float(np.mean(excess > 0)) if len(excess) else float("nan"),
    }
    return metrics, equity
