"""Map model scores to the six rating buckets by cross-sectional rank per date.

Bucketing happens strictly *within* each rebalance date (never pooled across
time), so a rating reflects relative attractiveness on that date only.
"""

from __future__ import annotations

import pandas as pd

from etf_intel.common.config import RatingsConfig
from etf_intel.common.types import Cols


def _bucket_counts(n: int, fractions: list[float]) -> list[int]:
    """Turn target fractions into integer counts summing to ``n``."""
    counts = [int(round(f * n)) for f in fractions]
    diff = n - sum(counts)
    # Absorb rounding error into the largest bucket to keep the total exact.
    idx = counts.index(max(counts)) if counts else 0
    counts[idx] += diff
    counts = [max(0, c) for c in counts]
    # Final guard: fix any residual by trimming/padding the largest bucket.
    residual = n - sum(counts)
    if residual != 0 and counts:
        idx = counts.index(max(counts))
        counts[idx] += residual
    return counts


def assign_ratings(scored: pd.DataFrame, ratings: RatingsConfig) -> pd.DataFrame:
    """Assign a rank and rating bucket to each ticker within each date.

    Args:
        scored: Frame with ``date``, ``ticker`` and a ``score`` column.
        ratings: Bucket fraction configuration (best to worst).

    Returns:
        A copy of ``scored`` with integer ``rank`` (1 = best) and string
        ``rating`` columns added, sorted by ``(date, rank)``.
    """
    pairs = ratings.ordered_fractions()
    rating_values = [r.value for r, _ in pairs]
    fractions = [f for _, f in pairs]

    out_frames: list[pd.DataFrame] = []
    for _, group in scored.groupby(Cols.DATE, sort=True):
        g = group.sort_values(Cols.SCORE, ascending=False).reset_index(drop=True)
        n = len(g)
        counts = _bucket_counts(n, fractions)
        labels: list[str] = []
        for value, c in zip(rating_values, counts, strict=True):
            labels.extend([value] * c)
        g[Cols.RATING] = labels[:n]
        g[Cols.RANK] = range(1, n + 1)
        out_frames.append(g)

    out = pd.concat(out_frames, ignore_index=True)
    return out.sort_values([Cols.DATE, Cols.RANK]).reset_index(drop=True)
