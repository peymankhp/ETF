"""ETF Intel — AI ETF research platform (MVP).

A point-in-time-correct, reproducible pipeline that ingests ETF and macro data,
engineers causal features, trains a model, walk-forward backtests it, and ranks
ETFs into six rating buckets with SHAP explanations.
"""

import contextlib
import sys as _sys

# Windows OpenMP load-order workaround: LightGBM's bundled OpenMP runtime must be
# loaded before numpy/scipy load theirs, otherwise the native library crashes with
# an access violation inside LGBM_DatasetSetField. Pre-importing lightgbm here (at
# package import, before any numpy usage) makes the whole pipeline safe on Windows.
if _sys.platform == "win32":  # pragma: no cover - platform-specific
    with contextlib.suppress(Exception):  # lightgbm optional for config-only usage
        import lightgbm as _lightgbm  # noqa: F401

__version__ = "0.1.0"
