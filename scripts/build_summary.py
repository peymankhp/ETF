"""Write a compact bot-facing summary (data/reports/summary.json).

Consumed by the on-demand Telegram bot (bot/bot.py). Contains just the current
buy-side ranking and the backtest headline — small and public-safe.

Usage:
    python scripts/build_summary.py
"""

from __future__ import annotations

from _common import load_context
from etf_intel.common.logging import get_logger
from etf_intel.common.types import Cols
from etf_intel.datastore import Paths

logger = get_logger("build_summary")


def main() -> None:
    """Build and store the compact summary JSON."""
    import pandas as pd

    _config, _settings, _universe, store = load_context()
    if not store.exists(Paths.LATEST_RANKING):
        logger.warning("No ranking found. Run the pipeline first.")
        return

    ranking = store.read_parquet(Paths.LATEST_RANKING).sort_values(Cols.RANK)
    as_of = pd.Timestamp(ranking[Cols.DATE].max())
    buys = ranking[ranking[Cols.RATING].isin(["Strong Buy", "Buy"])]

    summary: dict[str, object] = {
        "project": "ETF Intel",
        "as_of": f"{as_of:%Y-%m-%d}",
        "buys": [
            {
                "rank": int(r[Cols.RANK]),
                "ticker": str(r[Cols.TICKER]),
                "rating": str(r[Cols.RATING]),
            }
            for _, r in buys.iterrows()
        ],
    }
    if store.exists(Paths.METRICS):
        m = store.read_json(Paths.METRICS)
        s = m.get("strategy", {})
        summary["backtest"] = {
            "cagr": s.get("cagr"),
            "sharpe": s.get("sharpe"),
            "max_dd": s.get("max_dd"),
            "rank_ic": m.get("skill", {}).get("mean_xs_rank_ic"),
        }

    store.write_json(summary, "reports/summary.json")
    logger.info("Wrote summary.json (%d buys, as_of %s)", len(buys), summary["as_of"])


if __name__ == "__main__":
    main()
