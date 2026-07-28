"""Prediction-quality metrics (information coefficient, rank IC, AUC, ...)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error over aligned, non-null pairs."""
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return float("nan")
    err = y_true[mask].to_numpy() - y_pred[mask].to_numpy()
    return float(np.sqrt(np.mean(err**2)))


def information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Pearson correlation between predictions and realised target."""
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(y_true[mask], y_pred[mask])[0, 1])


def rank_information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Spearman (rank) correlation between predictions and realised target."""
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(y_true[mask].rank(), y_pred[mask].rank())[0, 1])


def auc(y_label: pd.Series, prob: pd.Series) -> float:
    """ROC AUC of a binary outperformance label vs predicted probability."""
    from sklearn.metrics import roc_auc_score

    mask = y_label.notna() & prob.notna()
    if mask.sum() < 2 or y_label[mask].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_label[mask], prob[mask]))
