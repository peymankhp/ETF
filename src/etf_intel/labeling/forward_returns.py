"""Forward-looking return labels and the excess-vs-benchmark target.

A label at date ``T`` deliberately uses bars after ``T`` (``index_{T+h}/index_T``).
This is legitimate *only* for labels, never for features. The leakage test's
positive control asserts these labels DO change when future prices change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_intel.common.config import AppConfig, Universe
from etf_intel.common.prices import total_return_index
from etf_intel.common.types import Cols

TARGET_COL = "target"
LABEL_OUTPERFORM_COL = "label_outperform"
EXCESS_COL = "excess_target"


def build_labels(market_df: pd.DataFrame, universe: Universe, config: AppConfig) -> pd.DataFrame:
    """Build forward-return labels for every horizon plus the model target.

    Args:
        market_df: Raw long market snapshot.
        universe: ETF universe (provides the benchmark).
        config: Application config (horizons + target definition).

    Returns:
        Long frame ``[date, ticker, <fwd_*>, excess_target, target, label_outperform]``.
        Rows whose forward window extends beyond available data have NaN labels
        (kept, so the latest date is still available for prediction).
    """
    horizons = config.horizons
    target_name = config.target.horizon
    if target_name not in horizons:
        raise ValueError(f"target horizon {target_name!r} not in horizons {list(horizons)}")

    market = market_df.sort_values([Cols.TICKER, Cols.DATE]).reset_index(drop=True)

    per_ticker: list[pd.DataFrame] = []
    for ticker, bars in market.groupby(Cols.TICKER, sort=False):
        bars = bars.reset_index(drop=True)
        idx = total_return_index(bars)
        cols: dict[str, object] = {
            Cols.DATE: bars[Cols.DATE].to_numpy(),
            Cols.TICKER: ticker,
        }
        for name, h in horizons.items():
            cols[name] = (idx.shift(-h) / idx - 1.0).to_numpy()
        per_ticker.append(pd.DataFrame(cols))

    labels = pd.concat(per_ticker, ignore_index=True)

    # Excess forward return vs the benchmark at the target horizon.
    bench_fwd = (
        labels[labels[Cols.TICKER] == universe.benchmark]
        .set_index(Cols.DATE)[target_name]
        .rename("bench_fwd")
    )
    aligned_bench = labels[Cols.DATE].map(bench_fwd)
    labels[EXCESS_COL] = labels[target_name] - aligned_bench

    if config.target.kind == "excess_vs_benchmark":
        labels[TARGET_COL] = labels[EXCESS_COL]
    else:
        labels[TARGET_COL] = labels[target_name]

    labels[LABEL_OUTPERFORM_COL] = (labels[EXCESS_COL] > 0).astype("float64")
    labels.loc[labels[EXCESS_COL].isna(), LABEL_OUTPERFORM_COL] = np.nan

    return labels.sort_values([Cols.DATE, Cols.TICKER]).reset_index(drop=True)
