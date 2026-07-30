"""Stress tests: crisis-window performance and sub-period skill stability.

Answers "can I trust these numbers?" beyond a single headline: how did the
strategy behave in named market crises, and is its cross-sectional skill stable
across time rather than driven by one lucky regime?

Honest limitation: the universe is *today's* surviving ETFs (survivorship bias),
and the walk-forward only starts once ~2y of history exists, so pre-~2013 crises
(2008 GFC) are outside the testable window. A true point-in-time universe needs
vintaged membership data (a v3+/paid-data item).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_intel.backtest.metrics import PERIODS_PER_YEAR, perf_stats
from etf_intel.common.types import Cols
from etf_intel.labeling import TARGET_COL

# Named crisis windows (only those overlapping the backtest are reported).
CRISIS_WINDOWS: list[tuple[str, str, str]] = [
    ("2015-16 selloff", "2015-08-01", "2016-02-29"),
    ("2018 Q4 selloff", "2018-10-01", "2018-12-31"),
    ("2020 COVID crash", "2020-02-01", "2020-04-30"),
    ("2022 rate shock", "2022-01-01", "2022-10-31"),
]


def _window_perf(
    equity: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, float] | None:
    sub = equity[(equity[Cols.DATE] >= start) & (equity[Cols.DATE] <= end)]
    if sub.empty:
        return None
    strat = float((1.0 + sub["strategy_return"]).prod() - 1.0)
    bench = float((1.0 + sub["benchmark_return"]).prod() - 1.0)
    eq = (1.0 + sub["strategy_return"]).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "n_periods": int(len(sub)),
        "strategy_return": strat,
        "benchmark_return": bench,
        "excess": strat - bench,
        "strategy_maxdd": max_dd,
    }


def crisis_performance(equity: pd.DataFrame) -> pd.DataFrame:
    """Strategy vs benchmark over each crisis window that overlaps the backtest."""
    rows: list[dict[str, object]] = []
    for name, start, end in CRISIS_WINDOWS:
        perf = _window_perf(equity, pd.Timestamp(start), pd.Timestamp(end))
        if perf is not None:
            rows.append({"window": name, **perf})
    return pd.DataFrame(rows)


def mean_xs_rank_ic(predictions: pd.DataFrame) -> float:
    """Mean per-date cross-sectional rank IC of score vs realised target."""
    valid = predictions.dropna(subset=[TARGET_COL, Cols.SCORE])
    ics: list[float] = []
    for _, g in valid.groupby(Cols.DATE):
        if len(g) >= 3 and g[Cols.SCORE].nunique() > 1 and g[TARGET_COL].nunique() > 1:
            ics.append(float(np.corrcoef(g[Cols.SCORE].rank(), g[TARGET_COL].rank())[0, 1]))
    return float(np.mean(ics)) if ics else float("nan")


def subperiod_stability(
    equity: pd.DataFrame, predictions: pd.DataFrame, n_periods: int = 3
) -> pd.DataFrame:
    """Split the backtest into ``n`` equal time slices; report skill + Sharpe in each.

    A strategy whose edge is real should show positive rank-IC and Sharpe across
    *all* slices, not just one lucky window.
    """
    if equity.empty:
        return pd.DataFrame()
    dates = np.array_split(np.sort(equity[Cols.DATE].unique()), n_periods)
    rows: list[dict[str, object]] = []
    for i, chunk in enumerate(dates, start=1):
        if len(chunk) == 0:
            continue
        lo, hi = chunk.min(), chunk.max()
        eq = equity[(equity[Cols.DATE] >= lo) & (equity[Cols.DATE] <= hi)]
        preds = predictions[(predictions[Cols.DATE] >= lo) & (predictions[Cols.DATE] <= hi)]
        strat = perf_stats(eq["strategy_return"], PERIODS_PER_YEAR)
        bench = perf_stats(eq["benchmark_return"], PERIODS_PER_YEAR)
        rows.append(
            {
                "slice": f"{i}/{n_periods}",
                "from": pd.Timestamp(lo).strftime("%Y-%m"),
                "to": pd.Timestamp(hi).strftime("%Y-%m"),
                "strategy_cagr": strat["cagr"],
                "strategy_sharpe": strat["sharpe"],
                "benchmark_sharpe": bench["sharpe"],
                "mean_xs_rank_ic": mean_xs_rank_ic(preds),
            }
        )
    return pd.DataFrame(rows)
