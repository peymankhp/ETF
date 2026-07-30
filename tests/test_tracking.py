"""Tests for the MLflow tracking helpers (pure flatten logic)."""

from __future__ import annotations

from etf_intel.common.config import AppConfig
from etf_intel.tracking.mlflow_tracker import _flatten_metrics, _flatten_params


def test_flatten_params_has_key_knobs() -> None:
    params = _flatten_params(AppConfig())
    for key in ("target_horizon", "model_kind", "rebalance", "portfolio_scheme", "no_trade_bands"):
        assert key in params


def test_flatten_metrics_flattens_nested_groups() -> None:
    metrics = {
        "n_periods": 100,
        "avg_turnover": 0.5,
        "strategy": {"cagr": 0.17, "sharpe": 1.03},
        "benchmark": {"sharpe": 1.12},
        "skill": {"rank_ic": 0.11, "auc": 0.53},
        "scheme": "equal",  # non-numeric -> dropped
    }
    flat = _flatten_metrics(metrics)
    assert flat["strategy_cagr"] == 0.17
    assert flat["benchmark_sharpe"] == 1.12
    assert flat["skill_rank_ic"] == 0.11
    assert flat["n_periods"] == 100.0
    assert "scheme" not in flat  # strings are not logged as metrics
