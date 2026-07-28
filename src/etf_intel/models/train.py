"""Train the LightGBM baseline: a return-score regressor + a calibrated classifier.

The regressor predicts the (excess) forward-return target; the classifier predicts
the probability of outperforming the benchmark, isotonically calibrated when the
sample is large and balanced enough (falls back to raw probabilities otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from etf_intel.common.config import AppConfig
from etf_intel.labeling import LABEL_OUTPERFORM_COL, TARGET_COL

MIN_CLASSIFIER_ROWS = 50
MIN_CLASS_COUNT_FOR_CALIBRATION = 6
CALIBRATION_CV = 3


@dataclass
class TrainedModel:
    """A fitted regressor + optional calibrated classifier and its feature list."""

    regressor: Any
    classifier: Any | None
    feature_cols: list[str]
    target_col: str = TARGET_COL
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def train_model(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    config: AppConfig,
    seed: int | None = None,
    calibrate: bool = True,
) -> TrainedModel:
    """Fit the regressor and (when feasible) a calibrated classifier.

    Args:
        dataset: Assembled feature+label frame (from ``assemble_training_frame``).
        feature_cols: Model input columns.
        config: Application config (holds model hyperparameters).
        seed: Optional seed override (defaults to ``config.seed``).
        calibrate: If ``True``, isotonically calibrate the classifier via CV. The
            walk-forward backtest sets this ``False`` to avoid tripling the
            classifier cost across hundreds of refits (its metrics use the score,
            not the probability).

    Returns:
        A :class:`TrainedModel`.
    """
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.calibration import CalibratedClassifierCV

    seed = config.seed if seed is None else seed
    params: dict[str, Any] = dict(config.model.params)
    params["random_state"] = seed

    fit_df = dataset.dropna(subset=[TARGET_COL])
    x = fit_df[feature_cols]
    y = fit_df[TARGET_COL]

    regressor = LGBMRegressor(**params)
    regressor.fit(x, y)

    classifier: Any | None = None
    y_cls = fit_df[LABEL_OUTPERFORM_COL]
    cmask = y_cls.notna()
    if cmask.sum() >= MIN_CLASSIFIER_ROWS and y_cls[cmask].nunique() == 2:
        cls_params = {k: v for k, v in params.items() if k != "objective"}
        base = LGBMClassifier(objective="binary", **cls_params)
        min_class = int(y_cls[cmask].astype(int).value_counts().min())
        x_cls, y_fit = x[cmask], y_cls[cmask].astype(int)
        if calibrate and min_class >= MIN_CLASS_COUNT_FOR_CALIBRATION:
            classifier = CalibratedClassifierCV(base, method="isotonic", cv=CALIBRATION_CV)
        else:
            classifier = base
        classifier.fit(x_cls, y_fit)

    return TrainedModel(
        regressor=regressor,
        classifier=classifier,
        feature_cols=list(feature_cols),
    )
