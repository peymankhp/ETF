"""Regressor factories (LightGBM / XGBoost) and a simple averaging ensemble.

Kept in its own module so the ensemble class is importable by path (picklable via
the model registry).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_lgbm_regressor(params: dict[str, Any]) -> Any:
    """Build a LightGBM regressor from the config params."""
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**params)


def build_xgb_regressor(params: dict[str, Any], seed: int) -> Any:
    """Build an XGBoost regressor, translating the shared params to XGBoost's API."""
    from xgboost import XGBRegressor

    max_depth = params.get("max_depth", -1)
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(params.get("n_estimators", 300)),
        learning_rate=float(params.get("learning_rate", 0.03)),
        max_depth=6 if max_depth in (-1, None) else int(max_depth),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=seed,
        n_jobs=int(params.get("n_jobs", 1)),
        verbosity=0,
    )


class EnsembleRegressor:
    """Averages the predictions of several fitted regressors (same target scale)."""

    def __init__(self, estimators: list[Any]):
        """Initialise with the sub-estimators to average.

        Args:
            estimators: Regressors implementing scikit-learn ``fit``/``predict``.
        """
        self.estimators = estimators

    def fit(self, x: pd.DataFrame, y: pd.Series) -> EnsembleRegressor:
        """Fit every sub-estimator on the same data."""
        for est in self.estimators:
            est.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Return the row-wise mean prediction across sub-estimators."""
        preds = np.column_stack(
            [np.asarray(est.predict(x), dtype=float) for est in self.estimators]
        )
        return preds.mean(axis=1)


def build_regressor(kind: str, params: dict[str, Any], seed: int) -> Any:
    """Return the regressor for a model kind: ``lightgbm`` | ``xgboost`` | ``ensemble``.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind == "lightgbm":
        return build_lgbm_regressor(params)
    if kind == "xgboost":
        return build_xgb_regressor(params, seed)
    if kind == "ensemble":
        return EnsembleRegressor([build_lgbm_regressor(params), build_xgb_regressor(params, seed)])
    raise ValueError(f"Unknown model kind: {kind!r}")
