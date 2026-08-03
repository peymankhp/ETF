"""Professional, visual HTML report for the ETF ranking (rendered to PDF).

Mirrors the SPAI report aesthetic (dark gradient header, KPI cards, tables) with a
teal/blue accent. Pure function: takes prepared data, returns an HTML string.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from etf_intel.common.types import Cols

_ACCENT = "#0f766e"  # teal
_DARK = "#0a0a2e"
_RATING_COLOR = {
    "Strong Buy": "#15803d",
    "Buy": "#22c55e",
    "Hold": "#a16207",
    "Reduce": "#ea580c",
    "Sell": "#dc2626",
    "Strong Sell": "#991b1b",
}


def _pct(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x * 100:.{d}f}%"


def _num(x: float | None, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def _kpi(value: str, label: str, sub: str, color: str) -> str:
    return f"""<td width="25%" style="background:#fff;border-radius:10px;padding:14px 8px;
        text-align:center;border-top:3px solid {color};box-shadow:0 2px 8px rgba(0,0,0,.07);">
      <div style="font-size:22px;font-weight:800;color:{color};letter-spacing:-.5px;">{value}</div>
      <div style="font-size:10px;font-weight:700;color:#888;text-transform:uppercase;
           letter-spacing:1px;margin-top:4px;">{label}</div>
      <div style="font-size:10px;color:#bbb;margin-top:2px;">{sub}</div></td>"""


def _rating_pill(rating: str) -> str:
    c = _RATING_COLOR.get(rating, "#666")
    return (
        f"<span style='background:{c}18;color:{c};border:1px solid {c};border-radius:10px;"
        f"padding:2px 8px;font-size:10px;font-weight:700;white-space:nowrap;'>{rating}</span>"
    )


def _prob_bar(p: float) -> str:
    w = max(0, min(100, p * 100))
    return (
        f"<div style='background:#eee;border-radius:6px;height:12px;width:80px;display:inline-block;'>"
        f"<div style='background:{_ACCENT};height:12px;border-radius:6px;width:{w:.0f}%;'></div></div>"
        f" <span style='font-size:10px;color:#888;'>{p * 100:.0f}%</span>"
    )


def _section(title: str, icon: str, content: str, bg: str = "#fff") -> str:
    return f"""<div style="background:{bg};border-radius:10px;padding:20px 22px 16px;
        margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,.07);">
      <div style="font-size:12px;font-weight:700;color:{_ACCENT};text-transform:uppercase;
           letter-spacing:1.5px;border-bottom:2px solid {_ACCENT};padding-bottom:8px;
           margin-bottom:14px;">{icon}&nbsp; {title}</div>{content}</div>"""


def _ranking_table(ranking: pd.DataFrame, limit: int = 16) -> str:
    head = (
        "<tr style='background:#f5f7fa;'>"
        "<th style='padding:8px 6px;text-align:left;font-size:10px;color:#666;'>#</th>"
        "<th style='padding:8px 6px;text-align:left;font-size:10px;color:#666;'>Ticker</th>"
        "<th style='padding:8px 6px;text-align:left;font-size:10px;color:#666;'>Rating</th>"
        "<th style='padding:8px 6px;text-align:right;font-size:10px;color:#666;'>Score</th>"
        "<th style='padding:8px 6px;text-align:left;font-size:10px;color:#666;'>P(beat SPY)</th></tr>"
    )
    rows = ""
    for i, (_, r) in enumerate(ranking.head(limit).iterrows()):
        bg = "#fafafa" if i % 2 else "#fff"
        rows += (
            f"<tr style='background:{bg};'>"
            f"<td style='padding:7px 6px;font-size:12px;color:#999;'>{int(r[Cols.RANK])}</td>"
            f"<td style='padding:7px 6px;font-size:13px;font-weight:700;color:#1a1a2e;'>{r[Cols.TICKER]}</td>"
            f"<td style='padding:7px 6px;'>{_rating_pill(str(r[Cols.RATING]))}</td>"
            f"<td style='padding:7px 6px;text-align:right;font-size:12px;'>{r[Cols.SCORE]:+.3f}</td>"
            f"<td style='padding:7px 6px;'>{_prob_bar(float(r[Cols.PROB_OUTPERFORM]))}</td></tr>"
        )
    return f"<table width='100%' style='border-collapse:collapse;'>{head}{rows}</table>"


def _holdings_card(h: dict[str, Any]) -> str:
    rows = ""
    for sym, name, wt in h.get("holdings", [])[:5]:
        rows += (
            f"<tr><td style='padding:2px 0;font-size:11px;color:#333;'>"
            f"<b>{sym}</b> <span style='color:#888;'>{name}</span></td>"
            f"<td style='padding:2px 0;text-align:right;font-size:11px;color:{_ACCENT};font-weight:700;'>"
            f"{wt * 100:.1f}%</td></tr>"
        )
    inside = (
        f"<table width='100%' style='margin-top:6px;'>{rows}</table>"
        if rows
        else f"<div style='font-size:11px;color:#888;margin-top:6px;font-style:italic;'>"
        f"{h.get('note', 'Physical / single-asset — no equity holdings.')}</div>"
    )
    return f"""<td width="50%" style="vertical-align:top;padding:6px;">
      <div style="background:#fff;border-radius:8px;padding:12px 14px;border-top:3px solid {_ACCENT};
           box-shadow:0 2px 8px rgba(0,0,0,.06);">
        <div style="font-size:14px;font-weight:800;color:#1a1a2e;">
          #{h["rank"]} {h["ticker"]} {_rating_pill(h["rating"])}</div>
        <div style="font-size:11px;color:#666;margin:2px 0 4px;">{h.get("name", "")}
          <span style="color:#aaa;"> · {h.get("category", "")}</span></div>
        <div style="font-size:10px;color:#999;text-transform:uppercase;letter-spacing:.5px;">Top holdings</div>
        {inside}</div></td>"""


def build_etf_html(
    as_of: pd.Timestamp,
    ranking: pd.DataFrame,
    metrics: dict[str, Any],
    holdings: list[dict[str, Any]],
    equity_png_b64: str | None = None,
    stress: pd.DataFrame | None = None,
) -> str:
    """Build the full ETF report HTML.

    Args:
        as_of: Ranking date.
        ranking: Scored+rated frame sorted by rank.
        metrics: Backtest metrics dict (strategy/benchmark/skill).
        holdings: Per-top-ETF dicts with rank/ticker/rating/name/category/holdings.
        equity_png_b64: base64 PNG of the equity curve (optional).
        stress: Sub-period stability frame (optional).

    Returns:
        The report HTML string.
    """
    s = metrics.get("strategy", {})
    b = metrics.get("benchmark", {})
    skill = metrics.get("skill", {})
    top = ranking.iloc[0] if len(ranking) else None
    date_str = f"{as_of:%B %d, %Y}"

    kpis = (
        _kpi(_pct(s.get("cagr")), "Strategy CAGR", f"SPY {_pct(b.get('cagr'))}", "#15803d")
        + "<td width='1%'></td>"
        + _kpi(_num(s.get("sharpe")), "Sharpe", f"SPY {_num(b.get('sharpe'))}", _ACCENT)
        + "<td width='1%'></td>"
        + _kpi(_pct(s.get("max_dd")), "Max Drawdown", f"SPY {_pct(b.get('max_dd'))}", "#dc2626")
        + "<td width='1%'></td>"
        + _kpi(
            f"{skill.get('mean_xs_rank_ic', float('nan')):+.3f}",
            "Skill (rank-IC)",
            "out-of-sample",
            "#7c3aed",
        )
    )
    kpi_row = f"<table width='100%'><tr>{kpis}</tr></table>"

    holdings_cards = ""
    pairs = holdings[:6]
    for j in range(0, len(pairs), 2):
        cell2 = _holdings_card(pairs[j + 1]) if j + 1 < len(pairs) else "<td width='50%'></td>"
        holdings_cards += f"<tr>{_holdings_card(pairs[j])}{cell2}</tr>"
    holdings_html = f"<table width='100%'>{holdings_cards}</table>"

    bt_rows = ""
    for name, sv, bv, fmt in [
        ("CAGR", s.get("cagr"), b.get("cagr"), _pct),
        ("Sharpe", s.get("sharpe"), b.get("sharpe"), _num),
        ("Sortino", s.get("sortino"), b.get("sortino"), _num),
        ("Max drawdown", s.get("max_dd"), b.get("max_dd"), _pct),
        ("Hit rate", s.get("hit_rate"), b.get("hit_rate"), _pct),
    ]:
        bt_rows += (
            f"<tr><td style='padding:5px 6px;font-size:12px;color:#444;'>{name}</td>"
            f"<td style='padding:5px 6px;text-align:right;font-size:12px;font-weight:700;'>{fmt(sv)}</td>"
            f"<td style='padding:5px 6px;text-align:right;font-size:12px;color:#888;'>{fmt(bv)}</td></tr>"
        )
    bt_table = (
        f"<table width='100%' style='border-collapse:collapse;'>"
        f"<tr style='background:#f5f7fa;'><th style='padding:7px 6px;text-align:left;font-size:10px;color:#666;'>Metric</th>"
        f"<th style='padding:7px 6px;text-align:right;font-size:10px;color:#666;'>Strategy</th>"
        f"<th style='padding:7px 6px;text-align:right;font-size:10px;color:#666;'>SPY</th></tr>{bt_rows}</table>"
    )
    equity_img = (
        f"<div style='margin-top:12px;'><img src='data:image/png;base64,{equity_png_b64}' "
        f"style='width:100%;border-radius:6px;'/></div>"
        if equity_png_b64
        else ""
    )

    stress_html = ""
    if stress is not None and not stress.empty:
        sr = (
            "<tr style='background:#f5f7fa;'>"
            "<th style='padding:6px;text-align:left;font-size:10px;color:#666;'>Period</th>"
            "<th style='padding:6px;text-align:right;font-size:10px;color:#666;'>Strat Sharpe</th>"
            "<th style='padding:6px;text-align:right;font-size:10px;color:#666;'>SPY Sharpe</th>"
            "<th style='padding:6px;text-align:right;font-size:10px;color:#666;'>Rank-IC</th></tr>"
        )
        for _, r in stress.iterrows():
            sr += (
                f"<tr><td style='padding:5px 6px;font-size:12px;'>{r['from']} – {r['to']}</td>"
                f"<td style='padding:5px 6px;text-align:right;font-size:12px;font-weight:700;'>{r['strategy_sharpe']:.2f}</td>"
                f"<td style='padding:5px 6px;text-align:right;font-size:12px;color:#888;'>{r['benchmark_sharpe']:.2f}</td>"
                f"<td style='padding:5px 6px;text-align:right;font-size:12px;color:{_ACCENT};font-weight:700;'>{r['mean_xs_rank_ic']:+.3f}</td></tr>"
            )
        stress_html = (
            f"<p style='font-size:11px;color:#888;margin:0 0 10px;'>The edge should persist "
            f"across time, not come from one lucky window.</p>"
            f"<table width='100%' style='border-collapse:collapse;'>{sr}</table>"
        )

    hero = ""
    if top is not None:
        hero = (
            f"<div style='margin-top:16px;padding:12px 16px;background:rgba(255,255,255,.08);"
            f"border-radius:8px;border-left:4px solid #2dd4bf;display:inline-block;'>"
            f"<span style='font-size:11px;color:#9fb;text-transform:uppercase;letter-spacing:1px;'>"
            f"Top pick</span><br><span style='font-size:30px;font-weight:900;color:#2dd4bf;'>"
            f"{top[Cols.TICKER]}</span> <span style='font-size:14px;color:#ccd;'>"
            f"{top[Cols.RATING]}</span></div>"
        )

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="font-family:Arial,sans-serif;max-width:720px;margin:auto;padding:16px;background:#eef2f5;color:#222;">
  <table width="100%"><tr>
    <td style="background:{_DARK};color:#2dd4bf;padding:9px 20px;border-radius:10px 10px 0 0;
        font-size:10.5px;letter-spacing:1.5px;">ETF INTEL — CROSS-SECTIONAL RANKING &nbsp;|&nbsp; Created by Peiman</td>
    <td style="background:{_DARK};color:#889;padding:9px 20px;border-radius:10px 10px 0 0;
        font-size:10.5px;text-align:right;">{date_str}</td></tr></table>
  <div style="background:linear-gradient(135deg,#0a0a2e,#0f3460 60%,#0f766e);color:#2dd4bf;padding:22px 24px 18px;">
    <div style="font-size:26px;font-weight:900;letter-spacing:.5px;">ETF Intelligence Report</div>
    <div style="font-size:12px;color:#aac;margin-top:5px;">
      LightGBM + XGBoost ensemble · 114 ETFs · 3-month cross-sectional ranking</div>
    {hero}</div>
  <div style="background:#eef2f5;padding:16px 0;">
    {_section("Performance Snapshot", "&#9889;", kpi_row)}
    {_section("Current Ranking", "&#128200;", _ranking_table(ranking))}
    {
        _section(
            "Inside the Top 5 — What You're Buying",
            "&#128269;",
            "<p style='font-size:11px;color:#888;margin:0 0 10px;'>Top holdings of each top-ranked ETF "
            "(source: fund data). Commodity/single-asset funds shown by mandate.</p>"
            + holdings_html,
        )
    }
    {_section("Walk-Forward Backtest (costs included)", "&#128202;", bt_table + equity_img)}
    {_section("Edge Stability — Stress Test", "&#129514;", stress_html) if stress_html else ""}
    {
        _section(
            "Methodology",
            "&#128214;",
            "<div style='font-size:12px;color:#444;line-height:1.7;'>"
            "Each ETF is scored by a LightGBM+XGBoost ensemble on point-in-time causal features "
            "(momentum, volatility, trend, relative strength, trailing dividend yield) to predict "
            "3-month excess return vs SPY. Ranked cross-sectionally into six buckets; the long book is "
            "the top bucket, equal-weighted with no-trade bands and 10 bps transaction costs. "
            "Everything is walk-forward and leakage-tested.</div>",
            bg="#f9f9f9",
        )
    }
  </div>
  <div style="text-align:center;font-size:10px;color:#aaa;line-height:1.8;padding:8px 0 20px;">
    Generated by <b>ETF Intel</b> — Created by <b>Peiman</b><br>
    Research signals only. <b>Not financial advice.</b> Past performance does not guarantee future results.</div>
</body></html>"""
