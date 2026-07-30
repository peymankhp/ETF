"""Detect and format week-over-week changes in the ETF ranking."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from etf_intel.common.types import Cols, Rating, rating_order

BUY_RATINGS = {Rating.STRONG_BUY.value, Rating.BUY.value}
_RATING_INDEX = {r.value: i for i, r in enumerate(rating_order())}  # lower = better


@dataclass
class RankingChanges:
    """Structured diff between a previous and current ranking."""

    new_strong_buys: list[str] = field(default_factory=list)
    new_buys: list[str] = field(default_factory=list)
    dropped_from_buys: list[str] = field(default_factory=list)
    upgrades: list[tuple[str, str, str]] = field(default_factory=list)  # (ticker, from, to)
    downgrades: list[tuple[str, str, str]] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return True if nothing changed."""
        return not (
            self.new_strong_buys
            or self.new_buys
            or self.dropped_from_buys
            or self.upgrades
            or self.downgrades
        )


def _rating_map(ranking: pd.DataFrame) -> dict[str, str]:
    return dict(zip(ranking[Cols.TICKER], ranking[Cols.RATING], strict=True))


def ranking_changes(previous: pd.DataFrame, current: pd.DataFrame) -> RankingChanges:
    """Compute the change set between two rankings.

    Args:
        previous: The prior ranking frame (``ticker``, ``rating``).
        current: The latest ranking frame (``ticker``, ``rating``).

    Returns:
        A :class:`RankingChanges` describing new/dropped buys and rating moves.
    """
    prev = _rating_map(previous)
    cur = _rating_map(current)

    def buys(m: dict[str, str]) -> set[str]:
        return {t for t, r in m.items() if r in BUY_RATINGS}

    def strong(m: dict[str, str]) -> set[str]:
        return {t for t, r in m.items() if r == Rating.STRONG_BUY.value}

    changes = RankingChanges(
        new_strong_buys=sorted(strong(cur) - strong(prev)),
        new_buys=sorted(buys(cur) - buys(prev)),
        dropped_from_buys=sorted(buys(prev) - buys(cur)),
    )
    for ticker in sorted(set(prev) & set(cur)):
        before, after = prev[ticker], cur[ticker]
        if before == after:
            continue
        if _RATING_INDEX[after] < _RATING_INDEX[before]:
            changes.upgrades.append((ticker, before, after))
        else:
            changes.downgrades.append((ticker, before, after))
    return changes


def format_alert(changes: RankingChanges, as_of: pd.Timestamp) -> str:
    """Render a short human-readable alert body from a change set."""
    if changes.is_empty():
        return f"ETF Intel ({as_of:%Y-%m-%d}): no ranking changes since last run."
    lines = [f"ETF Intel — ranking changes ({as_of:%Y-%m-%d})", ""]
    if changes.new_strong_buys:
        lines.append(f"🟢 New Strong Buy: {', '.join(changes.new_strong_buys)}")
    if changes.new_buys:
        lines.append(f"➕ New to buy-side: {', '.join(changes.new_buys)}")
    if changes.dropped_from_buys:
        lines.append(f"➖ Dropped from buy-side: {', '.join(changes.dropped_from_buys)}")
    if changes.upgrades:
        lines.append("⬆️ Upgrades: " + ", ".join(f"{t} {a}→{b}" for t, a, b in changes.upgrades))
    if changes.downgrades:
        lines.append("⬇️ Downgrades: " + ", ".join(f"{t} {a}→{b}" for t, a, b in changes.downgrades))
    return "\n".join(lines)
