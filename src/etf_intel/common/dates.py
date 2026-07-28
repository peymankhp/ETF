"""Date helpers and point-in-time guards.

Centralising these keeps the ``as_of`` invariant (no data after ``T``) explicit
and testable rather than scattered across modules.
"""

from __future__ import annotations

import pandas as pd

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_WEEK = 5
TRADING_DAYS_PER_MONTH = 21


def to_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Normalise a date-like value to a tz-naive :class:`pandas.Timestamp`.

    Args:
        value: A date string or timestamp.

    Returns:
        A tz-naive, normalised timestamp (time component dropped).
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None) if ts.tz is not None else ts.tz_localize(None)
    return ts.normalize()


def assert_as_of(frame: pd.DataFrame, as_of: pd.Timestamp, date_col: str = "date") -> None:
    """Raise if any row in ``frame`` carries a date strictly after ``as_of``.

    This is a defensive point-in-time guard: call it before feeding a frame into
    feature computation to catch accidental future-data leakage early.

    Args:
        frame: Frame to check.
        as_of: The as-of date; no row may exceed it.
        date_col: Name of the date column.

    Raises:
        ValueError: If any row's date is after ``as_of``.
    """
    if frame.empty:
        return
    max_date = pd.to_datetime(frame[date_col]).max()
    if max_date > as_of:
        raise ValueError(
            f"Point-in-time violation: frame contains data at {max_date!r} "
            f"which is after as_of={as_of!r}."
        )
