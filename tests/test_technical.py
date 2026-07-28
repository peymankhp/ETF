"""Correctness tests for hand-rolled technical indicators and adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_intel.common.prices import total_return_index
from etf_intel.common.types import Cols
from etf_intel.features.technical import ma_distance, rolling_return, rsi


def test_rolling_return_known_values() -> None:
    s = pd.Series([100.0, 110.0, 121.0])
    r = rolling_return(s, 1)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.10)
    assert r.iloc[2] == pytest.approx(0.10)


def test_rsi_of_monotonic_series_is_high() -> None:
    s = pd.Series(np.linspace(100, 200, 60))
    assert rsi(s, 14).iloc[-1] > 95.0


def test_ma_distance_zero_on_flat_series() -> None:
    s = pd.Series([50.0] * 60)
    assert ma_distance(s, 20).iloc[-1] == pytest.approx(0.0)


def test_total_return_index_equals_close_without_actions() -> None:
    n = 50
    close = np.linspace(100, 150, n)
    bars = pd.DataFrame(
        {
            Cols.DATE: pd.bdate_range("2020-01-01", periods=n),
            Cols.CLOSE: close,
            Cols.DIVIDENDS: 0.0,
            Cols.SPLITS: 1.0,
        }
    )
    np.testing.assert_allclose(total_return_index(bars).to_numpy(), close)


def test_total_return_index_reinvests_dividend() -> None:
    bars = pd.DataFrame(
        {
            Cols.DATE: pd.bdate_range("2020-01-01", periods=3),
            Cols.CLOSE: [100.0, 100.0, 100.0],
            Cols.DIVIDENDS: [0.0, 5.0, 0.0],  # $5 dividend on day 2
            Cols.SPLITS: [1.0, 1.0, 1.0],
        }
    )
    idx = total_return_index(bars)
    # Flat price but a dividend => total-return index rises and stays elevated.
    assert idx.iloc[1] > idx.iloc[0]
    assert idx.iloc[2] == pytest.approx(idx.iloc[1])
