"""Build the visual ETF report as a PDF (holdings + backtest + stress + equity).

Fetches top-5 ETF holdings (yfinance fund data), builds the HTML report, and
renders it to data/reports/etf_report.pdf via Playwright/Chromium.

Usage:
    python scripts/build_etf_pdf.py
"""

from __future__ import annotations

from _common import load_context
from etf_intel.common.logging import get_logger
from etf_intel.common.types import Cols
from etf_intel.datastore import Paths

logger = get_logger("build_etf_pdf")


def _fetch_holdings(tickers: list[str]) -> dict[str, dict[str, object]]:
    """Return {ticker: {name, category, holdings:[(sym,name,wt)...], note}}."""
    import yfinance as yf

    out: dict[str, dict[str, object]] = {}
    for t in tickers:
        entry: dict[str, object] = {"name": t, "category": "", "holdings": [], "note": ""}
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}
            entry["name"] = info.get("longName", t)
            entry["category"] = info.get("category", "")
            th = tk.funds_data.top_holdings
            if th is not None and len(th):
                entry["holdings"] = [
                    (str(sym), str(row["Name"]), float(row["Holding Percent"]))
                    for sym, row in th.head(5).iterrows()
                ]
            else:
                entry["note"] = "Physical / single-asset fund — tracks its mandate directly."
        except Exception as exc:  # network / schema issues -> mandate note
            logger.warning("holdings fetch failed for %s: %s", t, exc)
            entry["note"] = "Holdings unavailable."
        out[t] = entry
    return out


def render_etf_pdf(store_root: str, out_path: str = "reports/etf_report.pdf") -> str | None:
    """Build the ETF report HTML and render it to a PDF; return the path or None."""
    import base64
    from pathlib import Path

    import pandas as pd

    from etf_intel.backtest.stress import subperiod_stability
    from etf_intel.datastore import DataStore
    from etf_intel.reporting.html_report import build_etf_html

    store = DataStore(store_root)
    if not store.exists(Paths.LATEST_RANKING):
        logger.warning("No ranking; run the pipeline first.")
        return None
    ranking = store.read_parquet(Paths.LATEST_RANKING).sort_values(Cols.RANK)
    as_of = pd.Timestamp(ranking[Cols.DATE].max())
    metrics = store.read_json(Paths.METRICS) if store.exists(Paths.METRICS) else {}

    top5 = ranking.head(5)
    holdings_raw = _fetch_holdings(top5[Cols.TICKER].tolist())
    holdings = [
        {
            "rank": int(r[Cols.RANK]),
            "ticker": str(r[Cols.TICKER]),
            "rating": str(r[Cols.RATING]),
            **holdings_raw.get(str(r[Cols.TICKER]), {}),
        }
        for _, r in top5.iterrows()
    ]

    equity_b64 = None
    png = Path(store_root) / Paths.EQUITY_CURVE_PNG
    if png.exists():
        equity_b64 = base64.b64encode(png.read_bytes()).decode()

    stress = None
    if store.exists(Paths.EQUITY_CURVE) and store.exists(Paths.PREDICTIONS):
        stress = subperiod_stability(
            store.read_parquet(Paths.EQUITY_CURVE), store.read_parquet(Paths.PREDICTIONS)
        )

    html = build_etf_html(as_of, ranking, metrics, holdings, equity_b64, stress)

    from etf_intel.reporting.pdf import html_to_pdf

    out = str(store.path(out_path))
    html_to_pdf(html, out)
    logger.info("ETF report PDF -> %s", out)
    return out


def main() -> None:
    """Build the ETF report PDF."""
    config, _settings, _universe, _store = load_context()
    render_etf_pdf(config.data.root)


if __name__ == "__main__":
    main()
