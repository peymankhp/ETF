"""Tests for config loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from etf_intel.common.config import AppConfig, RatingsConfig, Universe


def test_app_config_loads(app_config: AppConfig) -> None:
    assert app_config.seed == 42
    assert app_config.target.horizon in app_config.horizons
    assert set(app_config.macro_series) == set(app_config.features.macro_release_lag_days)


def test_rating_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        RatingsConfig(strong_buy=0.5, buy=0.5, hold=0.5, reduce=0.0, sell=0.0, strong_sell=0.0)


def test_default_ratings_sum_to_one() -> None:
    total = sum(f for _, f in RatingsConfig().ordered_fractions())
    assert total == pytest.approx(1.0)


def test_universe_requires_benchmark_in_tickers() -> None:
    with pytest.raises(ValidationError):
        Universe(benchmark="SPY", tickers=["QQQ", "IWM"])


def test_universe_rejects_duplicates() -> None:
    with pytest.raises(ValidationError):
        Universe(benchmark="SPY", tickers=["SPY", "SPY"])
