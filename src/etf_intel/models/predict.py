"""Prediction: map a feature matrix to a return score and outperformance prob."""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_intel.common.types import Cols
from etf_intel.models.train import TrainedModel


def predict(model: TrainedModel, features: pd.DataFrame) -> pd.DataFrame:
    """Score a feature matrix.

    Args:
        model: A fitted :class:`TrainedModel`.
        features: Frame containing at least ``model.feature_cols``.

    Returns:
        Frame (same index as ``features``) with ``score`` and ``prob_outperform``.
    """
    x = features[model.feature_cols]
    score = np.asarray(model.regressor.predict(x), dtype=float)
    if model.classifier is not None:
        prob = np.asarray(model.classifier.predict_proba(x)[:, 1], dtype=float)
    else:
        prob = np.full(len(x), 0.5, dtype=float)
    return pd.DataFrame({Cols.SCORE: score, Cols.PROB_OUTPERFORM: prob}, index=features.index)
