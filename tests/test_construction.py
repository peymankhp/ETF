"""Tests for risk-aware portfolio weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_intel.portfolio.construction import compute_weights


@pytest.fixture
def trailing_returns() -> pd.DataFrame:
    """Three assets with clearly different volatilities."""
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame(
        {
            "LOWVOL": rng.normal(0.0, 0.005, n),
            "MIDVOL": rng.normal(0.0, 0.015, n),
            "HIGHVOL": rng.normal(0.0, 0.040, n),
        }
    )


def test_weights_sum_to_one_and_nonnegative(trailing_returns: pd.DataFrame) -> None:
    for scheme in ("equal", "inverse_vol"):
        w = compute_weights(scheme, trailing_returns, max_weight=1.0)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()


def test_equal_scheme_is_uniform(trailing_returns: pd.DataFrame) -> None:
    w = compute_weights("equal", trailing_returns, max_weight=1.0)
    assert w.nunique() == 1
    assert w.iloc[0] == pytest.approx(1.0 / 3)


def test_inverse_vol_prefers_low_vol(trailing_returns: pd.DataFrame) -> None:
    w = compute_weights("inverse_vol", trailing_returns, max_weight=1.0)
    # Lower volatility -> larger weight.
    assert w["LOWVOL"] > w["MIDVOL"] > w["HIGHVOL"]


def test_max_weight_cap_is_respected(trailing_returns: pd.DataFrame) -> None:
    w = compute_weights("inverse_vol", trailing_returns, max_weight=0.4)
    assert w.max() <= 0.4 + 1e-9
    assert w.sum() == pytest.approx(1.0)


def test_single_and_empty_book() -> None:
    one = compute_weights("inverse_vol", pd.DataFrame({"AAA": [0.01, -0.01]}), 0.15)
    assert one.to_dict() == {"AAA": 1.0}
    empty = compute_weights("equal", pd.DataFrame(), 0.15)
    assert empty.empty


def test_unknown_scheme_raises(trailing_returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown portfolio scheme"):
        compute_weights("nope", trailing_returns, 0.15)
