"""Feature engineering: causal technical, cross-sectional, and macro features.

Every feature at date ``T`` is a pure function of information available at ``T``.
This package must never import :mod:`etf_intel.labeling` (enforced by
import-linter) so features cannot see the target.
"""

from etf_intel.features.adjust import total_return_index
from etf_intel.features.pipeline import build_features

__all__ = ["build_features", "total_return_index"]
