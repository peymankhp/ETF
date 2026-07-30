"""Tests for ranking-change detection and alert formatting."""

from __future__ import annotations

import pandas as pd

from etf_intel.alerting import format_alert, ranking_changes
from etf_intel.common.types import Cols


def _ranking(rows: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({Cols.TICKER: list(rows), Cols.RATING: list(rows.values())})


def test_ranking_changes_detects_moves() -> None:
    prev = _ranking({"AAA": "Buy", "BBB": "Hold", "CCC": "Strong Buy", "DDD": "Sell"})
    cur = _ranking({"AAA": "Strong Buy", "BBB": "Hold", "CCC": "Reduce", "DDD": "Buy"})

    ch = ranking_changes(prev, cur)
    assert ch.new_strong_buys == ["AAA"]
    assert "DDD" in ch.new_buys  # Sell -> Buy joins the buy-side
    assert "CCC" in ch.dropped_from_buys  # Strong Buy -> Reduce leaves buy-side
    assert ("AAA", "Buy", "Strong Buy") in ch.upgrades
    assert ("CCC", "Strong Buy", "Reduce") in ch.downgrades
    assert not ch.is_empty()


def test_no_changes_is_empty() -> None:
    r = _ranking({"AAA": "Buy", "BBB": "Hold"})
    ch = ranking_changes(r, r.copy())
    assert ch.is_empty()
    body = format_alert(ch, pd.Timestamp("2026-07-30"))
    assert "no ranking changes" in body


def test_format_alert_mentions_new_strong_buy() -> None:
    prev = _ranking({"AAA": "Hold"})
    cur = _ranking({"AAA": "Strong Buy"})
    body = format_alert(ranking_changes(prev, cur), pd.Timestamp("2026-07-30"))
    assert "AAA" in body and "Strong Buy" in body
