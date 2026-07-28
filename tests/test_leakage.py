"""The anti-leakage safety net.

Core idea: perturb prices strictly *after* a cutoff date and recompute. A
point-in-time-correct feature at or before the cutoff must be byte-identical,
because it cannot see the future. Labels, by contrast, MUST change — that is the
positive control proving the test has teeth.

If any feature ever starts using future data, ``test_features_are_causal`` fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_intel.common.config import AppConfig, Universe
from etf_intel.common.types import Cols
from etf_intel.features import build_features
from etf_intel.features.pipeline import feature_columns
from etf_intel.labeling import TARGET_COL, build_labels


def _perturb_future(market: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return a copy with all prices after ``cutoff`` multiplied by 1.5."""
    out = market.copy()
    future = out[Cols.DATE] > cutoff
    for col in (Cols.OPEN, Cols.HIGH, Cols.LOW, Cols.CLOSE):
        out.loc[future, col] = out.loc[future, col] * 1.5
    return out


def test_features_are_causal(
    market: pd.DataFrame, universe: Universe, app_config: AppConfig
) -> None:
    """Features at/before the cutoff are invariant to future price changes."""
    dates = np.sort(market[Cols.DATE].unique())
    cutoff = pd.Timestamp(dates[len(dates) // 2])

    feat_full = build_features(market, None, universe, app_config)
    feat_pert = build_features(_perturb_future(market, cutoff), None, universe, app_config)

    cols = feature_columns(feat_full)
    a = (
        feat_full[feat_full[Cols.DATE] <= cutoff]
        .sort_values([Cols.DATE, Cols.TICKER])
        .reset_index(drop=True)
    )
    b = (
        feat_pert[feat_pert[Cols.DATE] <= cutoff]
        .sort_values([Cols.DATE, Cols.TICKER])
        .reset_index(drop=True)
    )
    assert list(a[Cols.TICKER]) == list(b[Cols.TICKER])
    # Any leakage of future data would make at least one of these differ.
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-10)


def test_labels_do_use_future_positive_control(
    market: pd.DataFrame, universe: Universe, app_config: AppConfig
) -> None:
    """Forward-return labels near the cutoff MUST change (proves test sensitivity)."""
    dates = np.sort(market[Cols.DATE].unique())
    cutoff = pd.Timestamp(dates[len(dates) // 2])

    lab_full = (
        build_labels(market, universe, app_config)
        .sort_values([Cols.DATE, Cols.TICKER])
        .reset_index(drop=True)
    )
    lab_pert = (
        build_labels(_perturb_future(market, cutoff), universe, app_config)
        .sort_values([Cols.DATE, Cols.TICKER])
        .reset_index(drop=True)
    )

    # Labels within one horizon before the cutoff look past it and must differ.
    horizon = app_config.horizons[app_config.target.horizon]
    window_start = pd.Timestamp(dates[len(dates) // 2 - horizon])
    mask = (lab_full[Cols.DATE] >= window_start) & (lab_full[Cols.DATE] <= cutoff)
    assert not np.allclose(
        lab_full.loc[mask, TARGET_COL].fillna(0.0),
        lab_pert.loc[mask, TARGET_COL].fillna(0.0),
    )


def test_leakage_detector_catches_a_leaky_feature() -> None:
    """Sanity-check the methodology: a deliberately leaky feature is detected."""
    s = pd.Series(np.random.default_rng(0).normal(size=200).cumsum() + 100)
    cutoff = 100

    def causal_feature(x: pd.Series) -> pd.Series:
        return x.rolling(10).mean()  # trailing window -> causal

    def leaky_feature(x: pd.Series) -> pd.Series:
        return x.rolling(10, center=True).mean()  # peeks ahead -> leaky

    s_future = s.copy()
    s_future.iloc[cutoff + 1 :] *= 2.0

    # Causal feature is unchanged up to the cutoff.
    np.testing.assert_allclose(
        causal_feature(s).iloc[: cutoff + 1].to_numpy(),
        causal_feature(s_future).iloc[: cutoff + 1].to_numpy(),
        equal_nan=True,
    )
    # Leaky feature differs before the cutoff -> the detector would fail a real feature.
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            leaky_feature(s).iloc[: cutoff + 1].to_numpy(),
            leaky_feature(s_future).iloc[: cutoff + 1].to_numpy(),
            equal_nan=True,
        )
