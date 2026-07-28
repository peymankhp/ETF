"""Walk-forward backtest -> predictions, metrics, and an equity curve.

Usage:
    python scripts/run_backtest.py
"""

from __future__ import annotations

from _common import load_context
from etf_intel.backtest import compute_backtest, run_walk_forward
from etf_intel.common.logging import get_logger
from etf_intel.datastore import Paths
from etf_intel.features import build_features
from etf_intel.features.pipeline import feature_columns
from etf_intel.labeling import build_labels

logger = get_logger("run_backtest")


def main() -> None:
    """Run the walk-forward backtest and persist its outputs."""
    config, settings, universe, store = load_context()

    if store.exists(Paths.FEATURES) and store.exists(Paths.LABELS):
        features = store.read_parquet(Paths.FEATURES)
        labels = store.read_parquet(Paths.LABELS)
    else:
        market = store.read_parquet(Paths.MARKET)
        macro = store.read_parquet(Paths.MACRO) if store.exists(Paths.MACRO) else None
        features = build_features(market, macro, universe, config)
        labels = build_labels(market, universe, config)

    fcols = feature_columns(features)
    logger.info("Running walk-forward backtest (%d features)", len(fcols))
    predictions = run_walk_forward(features, fcols, universe, config, labels=labels)
    metrics, equity = compute_backtest(predictions, universe, config)

    store.write_parquet(predictions, Paths.PREDICTIONS)
    store.write_json(metrics, Paths.METRICS)
    store.write_parquet(equity, Paths.EQUITY_CURVE)

    from etf_intel.reporting import save_equity_curve

    save_equity_curve(equity, store.path(Paths.EQUITY_CURVE_PNG))

    strat = metrics.get("strategy", {})
    logger.info(
        "Backtest: %d periods | strat CAGR=%.2f%% Sharpe=%.2f MaxDD=%.2f%% | excess hit=%.1f%%",
        metrics.get("n_periods", 0),
        100 * strat.get("cagr", float("nan")),
        strat.get("sharpe", float("nan")),
        100 * strat.get("max_dd", float("nan")),
        100 * metrics.get("excess_hit_rate", float("nan")),
    )


if __name__ == "__main__":
    main()
