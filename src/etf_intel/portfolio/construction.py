"""Position weighting for the long book (equal / inverse-vol).

All schemes use only *trailing* returns (up to the rebalance date), so weights are
point-in-time correct. A per-position cap is applied and the excess redistributed.

Note: HRP via PyPortfolioOpt is deferred — its scipy clustering path hard-crashed
the interpreter (native fault) on the real backtest data on this platform, and it
underperformed equal-weight on this signal anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SchemeName = str


def _cap_weights(weights: pd.Series, max_weight: float) -> pd.Series:
    """Cap each weight at ``max_weight`` and redistribute the excess; renormalise."""
    w = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    w = w / total
    if max_weight >= 1.0 or len(w) <= 1:
        return w
    for _ in range(100):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if not under.any() or w[under].sum() <= 0:
            break
        w[under] = w[under] + excess * (w[under] / w[under].sum())
    return w / w.sum()


def _inverse_vol(trailing_returns: pd.DataFrame) -> pd.Series:
    """Weights proportional to 1 / trailing volatility (risk-parity-lite)."""
    vol = trailing_returns.std()
    inv = 1.0 / vol.replace(0.0, np.nan)
    if inv.notna().any():
        inv = inv.fillna(float(inv.mean()))
    else:
        inv = pd.Series(1.0, index=trailing_returns.columns)
    return inv / inv.sum()


def compute_weights(
    scheme: SchemeName, trailing_returns: pd.DataFrame, max_weight: float = 1.0
) -> pd.Series:
    """Compute long-only weights for the selected names.

    Args:
        scheme: One of ``equal``, ``inverse_vol``.
        trailing_returns: Daily returns (rows=dates up to the rebalance date,
            cols=selected tickers). Only trailing data — never future.
        max_weight: Per-position cap.

    Returns:
        A weight Series indexed by ticker, summing to 1 (empty if no names).

    Raises:
        ValueError: If ``scheme`` is unknown.
    """
    tickers = list(trailing_returns.columns)
    if not tickers:
        return pd.Series(dtype=float)
    if len(tickers) == 1:
        return pd.Series([1.0], index=tickers)

    if scheme == "equal":
        raw = pd.Series(1.0 / len(tickers), index=tickers)
    elif scheme == "inverse_vol":
        raw = _inverse_vol(trailing_returns)
    else:
        raise ValueError(f"Unknown portfolio scheme: {scheme!r}")

    return _cap_weights(raw, max_weight)
