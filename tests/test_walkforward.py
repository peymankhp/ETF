"""Smoke + invariant tests for the walk-forward backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_intel.backtest import compute_backtest, run_walk_forward
from etf_intel.backtest.walkforward import REALIZED_FWD
from etf_intel.common.config import AppConfig, Universe
from etf_intel.common.types import Cols
from etf_intel.features import build_features
from etf_intel.features.pipeline import feature_columns
from etf_intel.labeling import build_labels


def test_walk_forward_runs_and_ranks(
    market: pd.DataFrame, universe: Universe, fast_config: AppConfig
) -> None:
    features = build_features(market, None, universe, fast_config)
    labels = build_labels(market, universe, fast_config)
    fcols = feature_columns(features)

    preds = run_walk_forward(features, fcols, universe, fast_config, labels=labels)
    assert not preds.empty
    for col in (Cols.RATING, Cols.RANK, Cols.SCORE, Cols.PROB_OUTPERFORM, REALIZED_FWD):
        assert col in preds.columns
    # Probabilities are valid.
    assert preds[Cols.PROB_OUTPERFORM].between(0.0, 1.0).all()


def test_walk_forward_respects_min_history(
    market: pd.DataFrame, universe: Universe, fast_config: AppConfig
) -> None:
    features = build_features(market, None, universe, fast_config)
    labels = build_labels(market, universe, fast_config)
    fcols = feature_columns(features)
    preds = run_walk_forward(features, fcols, universe, fast_config, labels=labels)

    dates = np.sort(features[Cols.DATE].unique())
    horizon = fast_config.horizons[fast_config.target.horizon]
    gap = horizon + fast_config.backtest.embargo_days
    earliest_allowed = pd.Timestamp(dates[fast_config.backtest.train_min_days + gap])
    assert preds[Cols.DATE].min() >= earliest_allowed


def test_compute_backtest_produces_equity(
    market: pd.DataFrame, universe: Universe, fast_config: AppConfig
) -> None:
    features = build_features(market, None, universe, fast_config)
    labels = build_labels(market, universe, fast_config)
    fcols = feature_columns(features)
    preds = run_walk_forward(features, fcols, universe, fast_config, labels=labels)

    panel = (
        features.pivot(index=Cols.DATE, columns=Cols.TICKER, values=Cols.ADJ_CLOSE)
        .sort_index()
        .pct_change(fill_method=None)
    )
    metrics, equity = compute_backtest(preds, universe, fast_config, panel)
    assert metrics["n_periods"] >= 1
    assert metrics["scheme"] == fast_config.portfolio.scheme
    assert {"strategy_equity", "benchmark_equity"}.issubset(equity.columns)
    assert (equity["strategy_equity"] > 0).all()
