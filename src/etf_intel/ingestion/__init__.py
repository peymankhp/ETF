"""Data ingestion: provider interfaces and concrete adapters.

Adapters return **raw** records only (no feature logic). yfinance prices are
fetched *unadjusted* with corporate actions so adjustment can be recomputed
deterministically downstream (see ``features``).
"""

from etf_intel.ingestion.base import MacroDataProvider, MarketDataProvider
from etf_intel.ingestion.factory import get_macro_provider, get_market_provider

__all__ = [
    "MacroDataProvider",
    "MarketDataProvider",
    "get_macro_provider",
    "get_market_provider",
]
