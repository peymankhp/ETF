"""Shared bootstrap + context loading for the CLI scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def _add_src_to_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

from etf_intel.common import (  # noqa: E402
    AppConfig,
    Settings,
    Universe,
    load_config,
    load_settings,
    load_universe,
)
from etf_intel.common.logging import get_logger  # noqa: E402
from etf_intel.datastore import DataStore  # noqa: E402


def apply_source_override(settings: Settings, source: str | None) -> Settings:
    """Override provider sources from a ``--source`` flag."""
    if source == "synthetic":
        settings.market_source = "synthetic"
        settings.macro_source = "synthetic"
    elif source in ("live", "yfinance"):
        settings.market_source = "yfinance"
        settings.macro_source = "fred"
    return settings


def load_context(
    source: str | None = None,
) -> tuple[AppConfig, Settings, Universe, DataStore]:
    """Load config, settings, universe and datastore for a script run.

    Falls back to synthetic macro when FRED is selected but no API key is set.
    """
    log = get_logger("etf_intel.scripts")
    config = load_config()
    settings = apply_source_override(load_settings(), source)
    if settings.macro_source == "fred" and not settings.fred_api_key:
        log.warning("No FRED API key set; falling back to synthetic macro data.")
        settings.macro_source = "synthetic"
    universe = load_universe(config.universe_file)
    store = DataStore(config.data_root(settings))
    return config, settings, universe, store
