"""Build the ETF report PDF, merge it with the SPAI report PDF, and send one
combined PDF + one post to Telegram.

Usage:
    python scripts/send_combined_pdf.py --spai-pdf D:/Github/SPAI/data/processed/spai_report.pdf
"""

from __future__ import annotations

import argparse

from _common import load_context
from build_etf_pdf import render_etf_pdf
from etf_intel.alerting import TelegramAlertChannel
from etf_intel.common.logging import get_logger
from etf_intel.common.types import Cols
from etf_intel.datastore import Paths
from etf_intel.reporting.pdf import merge_pdfs

logger = get_logger("send_combined")


def _caption(store: object) -> str:
    import pandas as pd

    from etf_intel.datastore import DataStore

    assert isinstance(store, DataStore)
    rk = pd.read_parquet(store.root / Paths.LATEST_RANKING).sort_values(Cols.RANK)
    as_of = pd.Timestamp(rk[Cols.DATE].max())
    top5 = ", ".join(rk[Cols.TICKER].head(5))
    m = store.read_json(Paths.METRICS) if store.exists(Paths.METRICS) else {}
    s = m.get("strategy", {})
    parts = [
        f"📈 Weekly Intelligence — {as_of:%Y-%m-%d}",
        f"📊 ETF top 5: {top5}",
        f"   Strategy CAGR {s.get('cagr', float('nan')) * 100:.1f}% · "
        f"Sharpe {s.get('sharpe', float('nan')):.2f}",
    ]
    return "\n".join(parts) + "\n🥇 + SPAI Gold forecast (attached).\nNot financial advice."


def main() -> None:
    """Build ETF PDF, merge with SPAI PDF, and send the combined report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spai-pdf", default="D:/Github/SPAI/data/processed/spai_report.pdf")
    args = ap.parse_args()

    config, settings, _universe, store = load_context()
    etf_pdf = render_etf_pdf(config.data.root)
    if not etf_pdf:
        return

    pdfs = [etf_pdf]
    from pathlib import Path

    if Path(args.spai_pdf).exists():
        pdfs.append(args.spai_pdf)
    else:
        logger.warning("SPAI PDF not found (%s) — sending ETF only", args.spai_pdf)

    combined = merge_pdfs(pdfs, str(store.path("reports/combined_report.pdf")))
    caption = _caption(store)

    if settings.telegram_bot_token and settings.telegram_chat_id:
        TelegramAlertChannel(settings.telegram_bot_token, settings.telegram_chat_id).send_document(
            combined, caption
        )
        logger.info("Combined report sent to Telegram")
    else:
        logger.info("Telegram not configured; combined report at %s", combined)


if __name__ == "__main__":
    main()
