"""Forward-return labeling — the ONLY sanctioned use of future data.

Quarantined in its own module so the anti-leakage boundary is auditable:
``features`` may not import ``labeling`` (enforced by import-linter).
"""

from etf_intel.labeling.forward_returns import (
    LABEL_OUTPERFORM_COL,
    TARGET_COL,
    build_labels,
)

__all__ = ["LABEL_OUTPERFORM_COL", "TARGET_COL", "build_labels"]
