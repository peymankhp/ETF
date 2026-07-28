"""Weekly markdown report: ranking tables, metrics summary, and SHAP drivers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from etf_intel.common.types import Cols

DISCLAIMER = (
    "_This is a research and educational tool, not licensed financial advice. "
    "Ratings are signals for human review, never automated orders._"
)


def _fmt_pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:,.2f}%"


def _fmt_num(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:,.2f}"


def _ranking_table(ranking: pd.DataFrame) -> str:
    lines = [
        "| Rank | Ticker | Rating | Score | P(outperform) |",
        "|-----:|:-------|:-------|------:|--------------:|",
    ]
    for _, row in ranking.iterrows():
        lines.append(
            f"| {int(row[Cols.RANK])} | {row[Cols.TICKER]} | {row[Cols.RATING]} "
            f"| {_fmt_num(row[Cols.SCORE])} | {_fmt_pct(row[Cols.PROB_OUTPERFORM])} |"
        )
    return "\n".join(lines)


def _metrics_table(metrics: dict[str, Any]) -> str:
    strat = metrics.get("strategy", {})
    bench = metrics.get("benchmark", {})
    rows = [
        (
            "CAGR",
            _fmt_pct(strat.get("cagr", float("nan"))),
            _fmt_pct(bench.get("cagr", float("nan"))),
        ),
        (
            "Sharpe",
            _fmt_num(strat.get("sharpe", float("nan"))),
            _fmt_num(bench.get("sharpe", float("nan"))),
        ),
        (
            "Sortino",
            _fmt_num(strat.get("sortino", float("nan"))),
            _fmt_num(bench.get("sortino", float("nan"))),
        ),
        (
            "Max drawdown",
            _fmt_pct(strat.get("max_dd", float("nan"))),
            _fmt_pct(bench.get("max_dd", float("nan"))),
        ),
        (
            "Hit rate",
            _fmt_pct(strat.get("hit_rate", float("nan"))),
            _fmt_pct(bench.get("hit_rate", float("nan"))),
        ),
    ]
    out = [
        "| Metric | Strategy | Benchmark |",
        "|:-------|---------:|----------:|",
    ]
    out += [f"| {name} | {s} | {b} |" for name, s, b in rows]
    out.append("")
    out.append(
        f"Backtest periods: **{metrics.get('n_periods', 0)}** · "
        f"Excess hit rate: **{_fmt_pct(metrics.get('excess_hit_rate', float('nan')))}**"
    )
    return "\n".join(out)


def _skill_table(skill: dict[str, Any]) -> str:
    rows = [
        ("Information coefficient (IC)", _fmt_num(skill.get("ic", float("nan")))),
        ("Rank IC", _fmt_num(skill.get("rank_ic", float("nan")))),
        ("Mean cross-sectional rank IC", _fmt_num(skill.get("mean_xs_rank_ic", float("nan")))),
        ("AUC (outperform vs SPY)", _fmt_num(skill.get("auc", float("nan")))),
    ]
    out = ["| Skill metric (out-of-sample) | Value |", "|:-----------------------------|------:|"]
    out += [f"| {name} | {val} |" for name, val in rows]
    out.append("")
    out.append("_IC/rank-IC near 0 ≈ no edge; AUC near 0.5 ≈ coin flip. Higher is better._")
    return "\n".join(out)


def _provenance_banner(provenance: dict[str, str] | None) -> str:
    if not provenance:
        return ""
    market = provenance.get("market_source", "?")
    macro = provenance.get("macro_source", "?")
    lines = [f"**Data provenance:** market = `{market}`, macro = `{macro}`."]
    synthetic = [name for name, src in (("market", market), ("macro", macro)) if src == "synthetic"]
    if synthetic:
        lines.append(
            f"> ⚠️ **{' and '.join(synthetic)} data is SYNTHETIC** — the metrics below are a "
            "mechanics demo, NOT a real trading signal."
        )
    return "\n".join(lines)


def _drivers_section(explanations: dict[str, list[tuple[str, float]]]) -> str:
    lines = ["## Why — top SHAP drivers", ""]
    for ticker, drivers in explanations.items():
        parts = ", ".join(f"`{f}` ({v:+.4f})" for f, v in drivers)
        lines.append(f"- **{ticker}**: {parts}")
    return "\n".join(lines)


def generate_markdown(
    as_of: pd.Timestamp,
    ranking: pd.DataFrame,
    metrics: dict[str, Any],
    explanations: dict[str, list[tuple[str, float]]] | None = None,
    provenance: dict[str, str] | None = None,
) -> str:
    """Render the weekly report as a markdown string.

    Args:
        as_of: The report's as-of date (latest ranked date).
        ranking: Latest scored+rated frame, sorted by rank.
        metrics: Backtest metrics dict from ``compute_backtest``.
        explanations: Optional ticker -> SHAP drivers for the top names.
        provenance: Optional ``{market_source, macro_source}`` for the data banner.

    Returns:
        The full markdown document.
    """
    top_buys = ranking[ranking[Cols.RATING].isin(["Strong Buy", "Buy"])]
    banner = _provenance_banner(provenance)
    sections = [
        f"# ETF Intel — Weekly Report ({as_of:%Y-%m-%d})",
        "",
        DISCLAIMER,
        "",
        *([banner, ""] if banner else []),
        "## Predictive skill (out-of-sample)",
        "",
        _skill_table(metrics.get("skill", {})),
        "",
        "## Backtest performance (walk-forward)",
        "",
        _metrics_table(metrics),
        "",
        "## Current ranking",
        "",
        _ranking_table(ranking),
        "",
        f"**Buy-side names:** {', '.join(top_buys[Cols.TICKER]) or 'none'}",
    ]
    if explanations:
        sections += ["", _drivers_section(explanations)]
    return "\n".join(sections) + "\n"
