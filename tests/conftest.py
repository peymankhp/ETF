"""Shared pytest fixtures: synthetic data, config, and a small universe."""

from __future__ import annotations

# Import lightgbm before numpy/scipy so its OpenMP runtime loads first (Windows
# crash workaround; harmless elsewhere). Order is deliberate; isort disabled here.
# isort: off
import lightgbm  # noqa: F401
from pathlib import Path

import pandas as pd
import pytest

from etf_intel.common.config import AppConfig, Universe, load_config
from etf_intel.ingestion.synthetic import SyntheticMacroProvider, SyntheticMarketProvider

# isort: on

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app_config() -> AppConfig:
    """The real application config from ``config/settings.yaml``."""
    return load_config(REPO_ROOT / "config" / "settings.yaml")


@pytest.fixture
def fast_config(app_config: AppConfig) -> AppConfig:
    """A shrunken config for fast backtest tests."""
    return app_config.model_copy(
        update={
            "model": app_config.model.model_copy(
                update={"params": {**app_config.model.params, "n_estimators": 40}}
            ),
            "backtest": app_config.backtest.model_copy(update={"train_min_days": 252}),
        }
    )


@pytest.fixture
def universe() -> Universe:
    """A small four-ETF universe (SPY benchmark + three synthetic names)."""
    return Universe(benchmark="SPY", tickers=["SPY", "AAA", "BBB", "CCC"])


@pytest.fixture
def market(universe: Universe) -> pd.DataFrame:
    """Four years of reproducible synthetic OHLCV for the small universe."""
    provider = SyntheticMarketProvider(seed=7)
    return provider.fetch(universe.tickers, "2015-01-01", "2019-01-01")


@pytest.fixture
def macro(app_config: AppConfig) -> pd.DataFrame:
    """Reproducible synthetic macro series matching the config."""
    provider = SyntheticMacroProvider(seed=7)
    return provider.fetch(list(app_config.macro_series.keys()), "2015-01-01", "2019-01-01")
