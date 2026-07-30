"""Tests for the stress-test module (crisis windows + sub-period stability)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_intel.backtest.stress import (
    crisis_performance,
    mean_xs_rank_ic,
    subperiod_stability,
)
from etf_intel.common.types import Cols
from etf_intel.labeling import TARGET_COL


def _equity(start: str = "2019-01-31", periods: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=periods, freq="ME")
    return pd.DataFrame(
        {
            Cols.DATE: dates,
            "strategy_return": rng.normal(0.01, 0.04, periods),
            "benchmark_return": rng.normal(0.008, 0.04, periods),
        }
    )


def test_crisis_performance_picks_overlapping_windows() -> None:
    out = crisis_performance(_equity())
    assert not out.empty
    # 2020 COVID and 2022 windows fall inside 2019-2022.
    assert "2020 COVID crash" in set(out["window"])
    assert {"strategy_return", "benchmark_return", "excess", "strategy_maxdd"}.issubset(out.columns)


def test_subperiod_stability_returns_n_slices() -> None:
    eq = _equity()
    preds = pd.DataFrame(
        {
            Cols.DATE: list(eq[Cols.DATE]) * 3,
            Cols.TICKER: ["A"] * len(eq) + ["B"] * len(eq) + ["C"] * len(eq),
            Cols.SCORE: np.random.default_rng(1).normal(size=len(eq) * 3),
            TARGET_COL: np.random.default_rng(2).normal(size=len(eq) * 3),
        }
    )
    out = subperiod_stability(eq, preds, n_periods=3)
    assert len(out) == 3
    assert {"strategy_sharpe", "benchmark_sharpe", "mean_xs_rank_ic"}.issubset(out.columns)


def test_mean_xs_rank_ic_perfect_alignment_is_one() -> None:
    # Two dates, 3 names each, score perfectly rank-aligned with target.
    rows = []
    for d in ("2020-01-31", "2020-02-29"):
        for i, t in enumerate("ABC"):
            rows.append({Cols.DATE: pd.Timestamp(d), Cols.TICKER: t, Cols.SCORE: i, TARGET_COL: i})
    ic = mean_xs_rank_ic(pd.DataFrame(rows))
    assert ic == 1.0
