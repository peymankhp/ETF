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
from etf_intel.labeling import TARGET_COL
from etf_intel.models.metrics import (
    auc,
    information_coefficient,
    rank_information_coefficient,
)
from etf_intel.portfolio.construction import compute_weights

PERIODS_PER_YEAR = 12  # monthly rebalance
PERIODS_PER_YEAR_BY_REBALANCE = {"monthly": 12, "quarterly": 4}


def _skill_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Out-of-sample predictive-skill metrics (the honest read of the model).

    Args:
        predictions: Scored walk-forward predictions with ``score``, ``target``
            (realised excess return), and ``prob_outperform``.

    Returns:
        Dict with information coefficient, rank IC, mean per-date cross-sectional
        rank IC, and outperformance AUC.
    """
    valid = predictions.dropna(subset=[TARGET_COL, Cols.SCORE])
    if valid.empty:
        return {k: float("nan") for k in ("ic", "rank_ic", "mean_xs_rank_ic", "auc")}

    xs_ics: list[float] = []
    for _, g in valid.groupby(Cols.DATE):
        if len(g) >= 3 and g[Cols.SCORE].nunique() > 1 and g[TARGET_COL].nunique() > 1:
            xs_ics.append(float(np.corrcoef(g[Cols.SCORE].rank(), g[TARGET_COL].rank())[0, 1]))

    binary = (valid[TARGET_COL] > 0).astype(float)
    return {
        "ic": information_coefficient(valid[TARGET_COL], valid[Cols.SCORE]),
        "rank_ic": rank_information_coefficient(valid[TARGET_COL], valid[Cols.SCORE]),
        "mean_xs_rank_ic": float(np.mean(xs_ics)) if xs_ics else float("nan"),
        "auc": auc(binary, valid[Cols.PROB_OUTPERFORM]),
    }


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
    predictions: pd.DataFrame,
    universe: Universe,
    config: AppConfig,
    returns_panel: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build the equity curve and metrics for the top-bucket strategy.

    The long book is weighted per ``config.portfolio.scheme`` using only trailing
    returns (point-in-time), capped per position, and charged a turnover-based
    transaction cost each rebalance.

    Args:
        predictions: Scored predictions from :func:`run_walk_forward`.
        universe: ETF universe (benchmark).
        config: Application config.
        returns_panel: Wide daily-returns frame (index=date, cols=ticker) used for
            risk weighting. If ``None``, the book is equal-weighted.

    Returns:
        A ``(metrics, equity_df)`` tuple. ``equity_df`` has columns
        ``date, strategy_return, benchmark_return, strategy_equity,
        benchmark_equity``.
    """
    top_rating = Rating.STRONG_BUY.value
    pcfg = config.portfolio
    rows: list[tuple[pd.Timestamp, float, float]] = []
    turnovers: list[float] = []
    prev_w: pd.Series = pd.Series(dtype=float)

    for d, day in predictions.groupby(Cols.DATE):
        if pcfg.no_trade_bands:
            # Hysteresis: buy names in the top entry band; keep already-held names
            # until they fall out of a wider exit band. Cuts turnover without giving
            # up monthly signal refresh.
            n = len(day)
            entry_rank = max(1, int(np.ceil(pcfg.entry_top_frac * n)))
            exit_rank = max(entry_rank, int(np.ceil(pcfg.exit_top_frac * n)))
            rank_by_ticker = day.set_index(Cols.TICKER)[Cols.RANK]
            entries = set(rank_by_ticker.index[rank_by_ticker <= entry_rank])
            retained = set(prev_w.index) & set(rank_by_ticker.index[rank_by_ticker <= exit_rank])
            held = entries | retained
            selected = day[day[Cols.TICKER].isin(held)]
        else:
            selected = day[day[Cols.RATING] == top_rating]
        if selected.empty:
            # Degenerate top bucket (e.g. tiny universe where 10% rounds to zero
            # names): fall back to the single best-ranked name so the book is held.
            selected = day[day[Cols.RANK] == day[Cols.RANK].min()]

        realised = selected.set_index(Cols.TICKER)[REALIZED_FWD].dropna()
        bench_series = day.loc[day[Cols.TICKER] == universe.benchmark, REALIZED_FWD]
        if realised.empty or bench_series.empty or pd.isna(bench_series.iloc[0]):
            continue
        names = list(realised.index)

        # Trailing daily returns strictly up to d (point-in-time) for weighting.
        if returns_panel is not None and not returns_panel.empty:
            trail = (
                returns_panel.loc[returns_panel.index <= d]
                .reindex(columns=names)
                .tail(pcfg.vol_lookback)
            )
        else:
            trail = pd.DataFrame(columns=names)

        if trail.shape[0] >= 2:
            weights = compute_weights(pcfg.scheme, trail, pcfg.max_weight)
        else:
            weights = pd.Series(1.0 / len(names), index=names)
        weights = weights.reindex(names).fillna(0.0)
        weights = (
            weights / weights.sum()
            if weights.sum() > 0
            else pd.Series(1.0 / len(names), index=names)
        )

        gross = float((weights * realised.reindex(weights.index)).sum())
        idx = weights.index.union(prev_w.index)
        turnover = float(
            (weights.reindex(idx).fillna(0.0) - prev_w.reindex(idx).fillna(0.0)).abs().sum()
        )
        net = gross - turnover * pcfg.cost_bps / 10_000.0

        rows.append((pd.Timestamp(d), net, float(bench_series.iloc[0])))
        turnovers.append(turnover)
        prev_w = weights

    equity = pd.DataFrame(rows, columns=[Cols.DATE, "strategy_return", "benchmark_return"])
    if not equity.empty:
        equity["strategy_equity"] = (1.0 + equity["strategy_return"]).cumprod()
        equity["benchmark_equity"] = (1.0 + equity["benchmark_return"]).cumprod()

    ppy = PERIODS_PER_YEAR_BY_REBALANCE.get(config.backtest.rebalance, PERIODS_PER_YEAR)
    strat = (
        perf_stats(equity["strategy_return"], ppy)
        if not equity.empty
        else perf_stats(pd.Series([]))
    )
    bench = (
        perf_stats(equity["benchmark_return"], ppy)
        if not equity.empty
        else perf_stats(pd.Series([]))
    )
    excess = (
        equity["strategy_return"] - equity["benchmark_return"]
        if not equity.empty
        else pd.Series([], dtype=float)
    )
    metrics: dict[str, Any] = {
        "n_periods": int(len(equity)),
        "rebalance": config.backtest.rebalance,
        "scheme": pcfg.scheme,
        "avg_turnover": float(np.mean(turnovers)) if turnovers else float("nan"),
        "cost_bps": pcfg.cost_bps,
        "strategy": strat,
        "benchmark": bench,
        "excess_hit_rate": float(np.mean(excess > 0)) if len(excess) else float("nan"),
        "skill": _skill_metrics(predictions),
    }
    return metrics, equity
