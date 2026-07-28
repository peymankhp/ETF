"""Assemble the modelling frame by joining features and labels.

Kept separate from training so the join logic (and its point-in-time keys) is
testable in isolation.
"""

from __future__ import annotations

import pandas as pd

from etf_intel.common.types import Cols


def assemble_training_frame(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Inner-join features and labels on ``(date, ticker)``.

    Args:
        features: Long feature frame (from ``features.build_features``).
        labels: Long label frame (from ``labeling.build_labels``).

    Returns:
        Merged frame containing feature columns, the target, and label columns.
        Rows keep NaN labels near the series end (used for prediction, not fit).
    """
    label_cols = [c for c in labels.columns if c not in (Cols.DATE, Cols.TICKER)]
    return features.merge(
        labels[[Cols.DATE, Cols.TICKER, *label_cols]],
        on=[Cols.DATE, Cols.TICKER],
        how="inner",
    )
