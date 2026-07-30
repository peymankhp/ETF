"""Typed configuration: secrets from ``.env`` and non-secret params from YAML.

Secrets (``Settings``) come from environment / ``.env`` via pydantic-settings.
Everything else (``AppConfig``) is loaded from ``config/settings.yaml`` so runs
are reproducible and diff-able.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from etf_intel.common.types import Rating

DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")


class Settings(BaseSettings):
    """Runtime secrets and source switches (from environment / ``.env``)."""

    model_config = SettingsConfigDict(
        env_prefix="ETF_INTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fred_api_key: str = ""
    data_dir: str | None = None
    market_source: str = "yfinance"
    macro_source: str = "fred"
    resend_api_key: str = ""  # Resend email API key (optional, for alert emails)
    report_recipient: str = ""  # email address to send ranking-change alerts to


class DataConfig(BaseModel):
    """Storage layout and ingestion window."""

    root: str = "./data"
    duckdb_file: str = "etf_intel.duckdb"
    start_date: str = "2010-01-01"
    end_date: str | None = None
    min_history_days: int = 252  # drop tickers with fewer than this many bars at ingest


class MacdConfig(BaseModel):
    """MACD periods."""

    fast: int = 12
    slow: int = 26
    signal: int = 9


class FeaturesConfig(BaseModel):
    """Technical / cross-sectional / macro feature parameters."""

    return_windows: list[int] = Field(default_factory=lambda: [5, 21, 63, 126, 252])
    vol_windows: list[int] = Field(default_factory=lambda: [21, 63])
    ma_windows: list[int] = Field(default_factory=lambda: [50, 200])
    rsi_window: int = 14
    macd: MacdConfig = Field(default_factory=MacdConfig)
    drawdown_window: int = 252
    rel_strength_window: int = 63
    yield_window: int = 252  # trailing window for the dividend-yield feature
    macro_release_lag_days: dict[str, int] = Field(default_factory=dict)
    include_macro: bool = True  # macro is constant per-date -> useless for x-sectional ranking


class TargetConfig(BaseModel):
    """Which horizon/label the model optimises."""

    horizon: str = "fwd_1m"
    kind: str = "excess_vs_benchmark"


class ModelConfig(BaseModel):
    """Model family and hyperparameters."""

    kind: str = "lightgbm"
    params: dict[str, Any] = Field(default_factory=dict)


class BacktestConfig(BaseModel):
    """Walk-forward backtest parameters."""

    rebalance: str = "monthly"
    train_min_days: int = 504
    embargo_days: int = 21
    top_bucket_only: bool = True
    retrain_every_months: int = 3  # retrain cadence (predict monthly, retrain quarterly)


class PortfolioConfig(BaseModel):
    """Portfolio construction for the backtest's long book."""

    scheme: str = "equal"  # equal | inverse_vol
    max_weight: float = 0.15  # cap per position (after which weight is redistributed)
    cost_bps: float = 10.0  # round-trip transaction cost per unit of turnover (bps)
    vol_lookback: int = 63  # trailing days used to estimate vol / covariance
    no_trade_bands: bool = False  # hysteresis: hold names until they leave a wider band
    entry_top_frac: float = 0.10  # buy names ranked in the top this fraction
    exit_top_frac: float = 0.25  # keep held names until they fall below this fraction


class RatingsConfig(BaseModel):
    """Cross-sectional bucket fractions, ordered best to worst (must sum to 1)."""

    strong_buy: float = 0.10
    buy: float = 0.20
    hold: float = 0.40
    reduce: float = 0.15
    sell: float = 0.10
    strong_sell: float = 0.05

    @model_validator(mode="after")
    def _check_sum(self) -> RatingsConfig:
        total = self.strong_buy + self.buy + self.hold + self.reduce + self.sell + self.strong_sell
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Rating fractions must sum to 1.0, got {total:.6f}.")
        return self

    def ordered_fractions(self) -> list[tuple[Rating, float]]:
        """Return (rating, fraction) pairs ordered best to worst."""
        return [
            (Rating.STRONG_BUY, self.strong_buy),
            (Rating.BUY, self.buy),
            (Rating.HOLD, self.hold),
            (Rating.REDUCE, self.reduce),
            (Rating.SELL, self.sell),
            (Rating.STRONG_SELL, self.strong_sell),
        ]


class AppConfig(BaseModel):
    """Top-level non-secret configuration loaded from ``settings.yaml``."""

    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    universe_file: str = "./config/universe.yaml"
    horizons: dict[str, int] = Field(default_factory=dict)
    target: TargetConfig = Field(default_factory=TargetConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    macro_series: dict[str, str] = Field(default_factory=dict)
    model: ModelConfig = Field(default_factory=ModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    ratings: RatingsConfig = Field(default_factory=RatingsConfig)

    def data_root(self, settings: Settings | None = None) -> Path:
        """Resolve the effective data root, honouring an env override.

        Args:
            settings: Optional settings whose ``data_dir`` overrides the YAML root.

        Returns:
            The data root directory as a :class:`~pathlib.Path`.
        """
        if settings is not None and settings.data_dir:
            return Path(settings.data_dir)
        return Path(self.data.root)


class Universe(BaseModel):
    """The ETF universe and benchmark."""

    benchmark: str
    tickers: list[str]

    @model_validator(mode="after")
    def _validate(self) -> Universe:
        if not self.tickers:
            raise ValueError("Universe must contain at least one ticker.")
        if len(self.tickers) != len(set(self.tickers)):
            raise ValueError("Universe contains duplicate tickers.")
        if self.benchmark not in self.tickers:
            raise ValueError(f"Benchmark {self.benchmark!r} must be present in the ticker list.")
        return self


def load_settings() -> Settings:
    """Load runtime secrets/switches from environment and ``.env``."""
    return Settings()


def load_config(path: str | Path = DEFAULT_SETTINGS_PATH) -> AppConfig:
    """Load and validate the application config from a YAML file.

    Args:
        path: Path to ``settings.yaml``.

    Returns:
        A validated :class:`AppConfig`.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)


def load_universe(path: str | Path) -> Universe:
    """Load and validate the ETF universe from a YAML file.

    Args:
        path: Path to ``universe.yaml``.

    Returns:
        A validated :class:`Universe`.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Universe.model_validate(raw)
