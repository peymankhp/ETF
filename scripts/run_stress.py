"""Stress-test the strategy: crisis-window performance + sub-period stability.

Reads the saved backtest (equity curve + predictions) and reports how the strategy
held up in market crises and whether its skill is stable across time. Writes a
markdown stress report.

Usage:
    python scripts/run_stress.py
"""

from __future__ import annotations

import math

from _common import load_context
from etf_intel.backtest.stress import crisis_performance, subperiod_stability
from etf_intel.common.logging import get_logger
from etf_intel.datastore import Paths

logger = get_logger("run_stress")

SURVIVORSHIP_NOTE = (
    "> **Note (survivorship bias):** These tests use *today's* surviving ETFs and the "
    "walk-forward starts once ~2y of history exists, so pre-~2013 crises (2008 GFC) "
    "are outside the testable window. A true point-in-time universe needs vintaged "
    "membership data."
)


def _fmt_pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x * 100:,.1f}%"


def main() -> None:
    """Compute crisis-window and sub-period stress tables and save a report."""
    _config, _settings, _universe, store = load_context()
    if not (store.exists(Paths.EQUITY_CURVE) and store.exists(Paths.PREDICTIONS)):
        logger.warning("No backtest outputs found. Run the backtest first.")
        return

    equity = store.read_parquet(Paths.EQUITY_CURVE)
    predictions = store.read_parquet(Paths.PREDICTIONS)

    crisis = crisis_performance(equity)
    stability = subperiod_stability(equity, predictions)

    lines = ["# ETF Intel — Stress Test", "", SURVIVORSHIP_NOTE, ""]
    lines += ["## Crisis windows (strategy vs SPY)", ""]
    if crisis.empty:
        lines.append("_No crisis windows overlap the backtest period._")
    else:
        lines += ["| Window | Strategy | SPY | Excess | Strat MaxDD |", "|:--|--:|--:|--:|--:|"]
        for _, r in crisis.iterrows():
            lines.append(
                f"| {r['window']} | {_fmt_pct(r['strategy_return'])} | "
                f"{_fmt_pct(r['benchmark_return'])} | {_fmt_pct(r['excess'])} | "
                f"{_fmt_pct(r['strategy_maxdd'])} |"
            )
    lines += ["", "## Sub-period stability (is the edge persistent?)", ""]
    lines += [
        "| Slice | From | To | Strat CAGR | Strat Sharpe | SPY Sharpe | Rank-IC |",
        "|:--|:--|:--|--:|--:|--:|--:|",
    ]
    for _, r in stability.iterrows():
        lines.append(
            f"| {r['slice']} | {r['from']} | {r['to']} | {_fmt_pct(r['strategy_cagr'])} | "
            f"{r['strategy_sharpe']:.2f} | {r['benchmark_sharpe']:.2f} | "
            f"{r['mean_xs_rank_ic']:+.3f} |"
        )

    report = "\n".join(lines) + "\n"
    path = store.path("reports/stress_report.md")
    path.write_text(report, encoding="utf-8")
    logger.info("Stress report written -> %s", path)
    print("\n" + report)


if __name__ == "__main__":
    main()
