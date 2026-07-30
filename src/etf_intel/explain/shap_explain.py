"""SHAP explanations for the return-score regressor.

Produces, for each scored row, the features that pushed the prediction up or down
the most — the human-readable "why" behind a rating.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from etf_intel.models.train import TrainedModel

Driver = tuple[str, float]


class ShapExplainer:
    """Wraps a SHAP ``TreeExplainer`` over a trained regressor."""

    def __init__(self, model: TrainedModel):
        """Initialise the explainer.

        Args:
            model: A fitted :class:`TrainedModel` (its regressor is explained).
        """
        import shap

        self.feature_cols = model.feature_cols
        # Unwrap an ensemble to its first (LightGBM) sub-model, which TreeExplainer
        # understands; the averaging wrapper itself is not a tree model.
        regressor = model.regressor
        if hasattr(regressor, "estimators"):
            regressor = regressor.estimators[0]
        self.explainer: Any = shap.TreeExplainer(regressor)

    def _shap_matrix(self, features: pd.DataFrame) -> np.ndarray:
        x = features[self.feature_cols]
        values = self.explainer.shap_values(x)
        if isinstance(values, list):  # some versions return a list per output
            values = values[0]
        return np.asarray(values, dtype=float)

    def top_drivers(self, features: pd.DataFrame, top_n: int = 5) -> dict[Any, list[Driver]]:
        """Return the top-``n`` absolute SHAP drivers for each row.

        Args:
            features: Frame containing the model's feature columns.
            top_n: Number of drivers to return per row.

        Returns:
            Mapping from the frame's index value to a list of
            ``(feature_name, signed_shap_value)`` pairs, largest magnitude first.
        """
        matrix = self._shap_matrix(features)
        result: dict[Any, list[Driver]] = {}
        for i, idx in enumerate(features.index):
            sv = matrix[i]
            order = np.argsort(-np.abs(sv))[:top_n]
            result[idx] = [(self.feature_cols[j], float(sv[j])) for j in order]
        return result
