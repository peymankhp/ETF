"""Ingest raw ETF OHLCV + macro series into versioned parquet snapshots.

Usage:
    python scripts/run_ingest.py [--source synthetic|live]
"""

from __future__ import annotations

import argparse

from _common import load_context
from etf_intel.common.logging import get_logger
from etf_intel.datastore import Paths
from etf_intel.ingestion import get_macro_provider, get_market_provider

logger = get_logger("run_ingest")


def main() -> None:
    """Fetch and snapshot market + macro data."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["synthetic", "live", "yfinance"], default=None)
    args = ap.parse_args()

    config, settings, universe, store = load_context(args.source)
    start, end = config.data.start_date, config.data.end_date

    logger.info("Ingesting %d tickers from %s", len(universe.tickers), settings.market_source)
    market = get_market_provider(settings, config.seed).fetch(universe.tickers, start, end)
    store.write_snapshot(
        market,
        Paths.MARKET,
        {"source": settings.market_source, "tickers": universe.tickers, "start": start, "end": end},
    )
    logger.info("Market snapshot: %d rows -> %s", len(market), Paths.MARKET)

    series = list(config.macro_series.keys())
    logger.info("Ingesting %d macro series from %s", len(series), settings.macro_source)
    macro = get_macro_provider(settings, config.seed).fetch(series, start, end)
    store.write_snapshot(macro, Paths.MACRO, {"source": settings.macro_source, "series": series})
    logger.info("Macro snapshot: %d rows -> %s", len(macro), Paths.MACRO)


if __name__ == "__main__":
    main()
