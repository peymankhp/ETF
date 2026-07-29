"""Walk-forward backtest engine with purge + embargo (Decision D).

For each monthly rebalance date ``T`` the model is retrained using only rows whose
entire forward-label window closes at least ``embargo`` trading days before ``T``.
Concretely, training is restricted to dates on or before the position
``T - (horizon + embargo)`` in the trading-day index, so no label window can
straddle the train/test boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_intel.common.config import AppConfig, Universe
from etf_intel.common.logging import get_logger
from etf_intel.common.types import Cols
from etf_intel.labeling import TARGET_COL, build_labels
from etf_intel.models.dataset import assemble_training_frame
from etf_intel.models.predict import predict
from etf_intel.models.train import train_model
from etf_intel.portfolio.ranking import assign_ratings

logger = get_logger(__name__)

REALIZED_FWD = "realized_fwd"
MIN_TRAIN_ROWS = 100


# Which forward-return column represents one holding period, per rebalance freq.
HOLDING_BY_REBALANCE = {"monthly": "fwd_1m", "quarterly": "fwd_3m"}


def _rebalance_dates(dates: pd.DatetimeIndex, rebalance: str) -> list[pd.Timestamp]:
    """Return the last available trading date of each rebalance period."""
    s = pd.Series(dates, index=dates)
    if rebalance == "quarterly":
        return list(s.groupby([dates.year, dates.quarter]).max())
    return list(s.groupby([dates.year, dates.month]).max())


def run_walk_forward(
    features: pd.DataFrame,
    feature_cols: list[str],
    universe: Universe,
    config: AppConfig,
    labels: pd.DataFrame | None = None,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the walk-forward backtest and return per-date scored predictions.

    Args:
        features: Long feature frame.
        feature_cols: Model input columns.
        universe: ETF universe.
        config: Application config.
        labels: Precomputed labels; if ``None``, built from ``market_df``.
        market_df: Raw market frame (required if ``labels`` is ``None``).

    Returns:
        Long frame with ``date, ticker, score, prob_outperform, rating, rank,
        realized_fwd, adj_close`` across all evaluated rebalance dates.
    """
    if labels is None:
        if market_df is None:
            raise ValueError("Provide either labels or market_df.")
        labels = build_labels(market_df, universe, config)

    dataset = assemble_training_frame(features, labels)
    target_name = config.target.horizon
    horizon_days = config.horizons[target_name]
    gap = horizon_days + config.backtest.embargo_days

    # The model predicts the target-horizon excess (TARGET_COL), but the realised
    # P&L per period is the holding-period return for the rebalance interval
    # (1m for monthly, 3m for quarterly) — kept separate so the equity curve stays
    # correct when the prediction horizon differs from the rebalance interval.
    holding_col = HOLDING_BY_REBALANCE.get(config.backtest.rebalance, "fwd_1m")
    if holding_col not in config.horizons:
        holding_col = target_name

    didx = pd.DatetimeIndex(np.sort(dataset[Cols.DATE].unique()))
    rebal_dates = _rebalance_dates(didx, config.backtest.rebalance)

    keep_cols = [Cols.DATE, Cols.TICKER, Cols.ADJ_CLOSE, holding_col, TARGET_COL]
    results: list[pd.DataFrame] = []

    # Predict every rebalance date, but retrain only every ``retrain_every_months``
    # (time-based, so it's correct at any rebalance frequency) to keep the walk-
    # forward tractable. A reused model was still trained purely on data before its
    # cutoff, so this introduces no lookahead — only mild staleness.
    retrain_every_days = max(1, config.backtest.retrain_every_months) * 30
    model = None
    last_retrain: pd.Timestamp | None = None

    for t in rebal_dates:
        i = int(didx.get_loc(t))
        cutoff_pos = i - gap
        if cutoff_pos < config.backtest.train_min_days:
            continue
        cutoff_date = didx[cutoff_pos]

        train_df = dataset[dataset[Cols.DATE] <= cutoff_date].dropna(subset=[TARGET_COL])
        if len(train_df) < MIN_TRAIN_ROWS:
            continue

        if model is None or (t - last_retrain).days >= retrain_every_days:
            model = train_model(train_df, feature_cols, config, calibrate=False)
            last_retrain = t

        assert model is not None  # set on the first eligible fold above
        test_df = dataset[dataset[Cols.DATE] == t]
        preds = predict(model, test_df)

        res = test_df[keep_cols].copy()
        res[Cols.SCORE] = preds[Cols.SCORE].to_numpy()
        res[Cols.PROB_OUTPERFORM] = preds[Cols.PROB_OUTPERFORM].to_numpy()
        res = res.rename(columns={holding_col: REALIZED_FWD})
        results.append(res)

    if not results:
        logger.warning("Walk-forward produced no folds (insufficient history).")
        return pd.DataFrame(
            columns=[
                Cols.DATE,
                Cols.TICKER,
                Cols.ADJ_CLOSE,
                REALIZED_FWD,
                TARGET_COL,
                Cols.SCORE,
                Cols.PROB_OUTPERFORM,
            ]
        )

    scored = pd.concat(results, ignore_index=True)
    return assign_ratings(scored, config.ratings)
