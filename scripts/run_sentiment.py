"""Live news-sentiment overlay for the current top-ranked ETFs.

Fetches current headlines (yfinance), scores them, and prints/saves a sentiment
table for the current buy-side names. This is a LIVE overlay only — it is not part
of the backtested model (free historical news makes sentiment un-backtestable).

Usage:
    python scripts/run_sentiment.py [--top N]
"""

from __future__ import annotations

import argparse

from _common import load_context
from etf_intel.common.logging import get_logger
from etf_intel.common.types import Cols
from etf_intel.datastore import Paths
from etf_intel.ingestion.sentiment import NewsSentimentProvider

logger = get_logger("run_sentiment")


def main() -> None:
    """Score current headlines for the top-ranked names and save the overlay."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=15, help="number of top-ranked names to score")
    args = ap.parse_args()

    _config, _settings, _universe, store = load_context()
    if not store.exists(Paths.LATEST_RANKING):
        logger.warning("No ranking found. Run the pipeline first.")
        return

    ranking = store.read_parquet(Paths.LATEST_RANKING)
    tickers = ranking.sort_values(Cols.RANK)[Cols.TICKER].head(args.top).tolist()
    logger.info("Fetching live headlines for %d names", len(tickers))

    scores = NewsSentimentProvider().fetch(tickers)
    scores = scores.sort_values("sentiment", ascending=False)
    store.write_parquet(scores, "reports/live_sentiment.parquet")

    logger.info("Live sentiment (higher = more positive headlines):")
    for _, row in scores.iterrows():
        logger.info(
            "  %-6s sentiment=%+.2f (%d headlines)",
            row[Cols.TICKER],
            row["sentiment"],
            int(row["n_headlines"]),
        )


if __name__ == "__main__":
    main()
