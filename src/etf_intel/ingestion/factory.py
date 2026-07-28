"""Factory helpers that pick a concrete provider from settings.

This is the single place that maps the ``ETF_INTEL_MARKET_SOURCE`` /
``ETF_INTEL_MACRO_SOURCE`` switches to an implementation, so callers stay
provider-agnostic.
"""

from __future__ import annotations

from etf_intel.common.config import Settings
from etf_intel.ingestion.base import MacroDataProvider, MarketDataProvider
from etf_intel.ingestion.synthetic import SyntheticMacroProvider, SyntheticMarketProvider


def get_market_provider(settings: Settings, seed: int = 42) -> MarketDataProvider:
    """Return the market-data provider selected by ``settings.market_source``.

    Args:
        settings: Runtime settings.
        seed: Seed for the synthetic provider (ignored otherwise).

    Returns:
        A concrete :class:`MarketDataProvider`.

    Raises:
        ValueError: If the configured source is unknown.
    """
    source = settings.market_source.lower()
    if source == "synthetic":
        return SyntheticMarketProvider(seed=seed)
    if source == "yfinance":
        from etf_intel.ingestion.yfinance_adapter import YFinanceMarketProvider

        return YFinanceMarketProvider()
    raise ValueError(f"Unknown market source: {settings.market_source!r}")


def get_macro_provider(settings: Settings, seed: int = 42) -> MacroDataProvider:
    """Return the macro-data provider selected by ``settings.macro_source``.

    Args:
        settings: Runtime settings.
        seed: Seed for the synthetic provider (ignored otherwise).

    Returns:
        A concrete :class:`MacroDataProvider`.

    Raises:
        ValueError: If the configured source is unknown.
    """
    source = settings.macro_source.lower()
    if source == "synthetic":
        return SyntheticMacroProvider(seed=seed)
    if source == "fred":
        from etf_intel.ingestion.fred_adapter import FredMacroProvider

        return FredMacroProvider(api_key=settings.fred_api_key)
    raise ValueError(f"Unknown macro source: {settings.macro_source!r}")
