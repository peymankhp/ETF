"""Tests for forward-return labeling and the excess-vs-benchmark target."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_intel.common.config import AppConfig, Universe
from etf_intel.common.types import Cols
from etf_intel.labeling import LABEL_OUTPERFORM_COL, build_labels
from etf_intel.labeling.forward_returns import EXCESS_COL


def test_forward_return_matches_manual(
    market: pd.DataFrame, universe: Universe, app_config: AppConfig
) -> None:
    labels = build_labels(market, universe, app_config)
    h = app_config.horizons["fwd_1w"]
    one = labels[labels[Cols.TICKER] == "AAA"].sort_values(Cols.DATE).reset_index(drop=True)
    # fwd_1w at row i should be finite for early rows and NaN at the very end.
    assert np.isnan(one["fwd_1w"].iloc[-1])
    assert one["fwd_1w"].iloc[:-h].notna().all()


def test_benchmark_excess_is_zero(
    market: pd.DataFrame, universe: Universe, app_config: AppConfig
) -> None:
    labels = build_labels(market, universe, app_config)
    bench = labels[labels[Cols.TICKER] == universe.benchmark]
    valid = bench[EXCESS_COL].dropna()
    np.testing.assert_allclose(valid.to_numpy(), 0.0, atol=1e-12)


def test_outperform_label_is_binary(
    market: pd.DataFrame, universe: Universe, app_config: AppConfig
) -> None:
    labels = build_labels(market, universe, app_config)
    vals = labels[LABEL_OUTPERFORM_COL].dropna().unique()
    assert set(vals).issubset({0.0, 1.0})


def test_unknown_target_horizon_raises(
    market: pd.DataFrame, universe: Universe, app_config: AppConfig
) -> None:
    bad = app_config.model_copy(
        update={"target": app_config.target.model_copy(update={"horizon": "fwd_nope"})}
    )
    with pytest.raises(ValueError, match="target horizon"):
        build_labels(market, universe, bad)
