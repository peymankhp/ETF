"""Deterministic, point-in-time price adjustment (Decision C).

The implementation lives in :mod:`etf_intel.common.prices` so that both
``features`` and ``labeling`` (sibling layers that may not import each other) can
share it. Re-exported here for ergonomic access from the features package.
"""

from __future__ import annotations

from etf_intel.common.prices import total_return_index

__all__ = ["total_return_index"]
