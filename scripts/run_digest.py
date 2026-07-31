"""Send the weekly ETF digest (ranking + performance) via email and Telegram.

Unlike run_alerts (which only fires on ranking *changes*), this ALWAYS sends —
it is the "every Monday" report. Safe to run anywhere the alert credentials are
configured; does nothing (but logs) if none are set.

Usage:
    python scripts/run_digest.py
"""

from __future__ import annotations

from _common import load_context
from etf_intel.alerting import ResendEmailAlertChannel, TelegramAlertChannel
from etf_intel.common.logging import get_logger
from etf_intel.common.types import Cols
from etf_intel.datastore import DataStore, Paths

logger = get_logger("run_digest")


def _build_body(store: DataStore) -> tuple[str, str]:
    """Compose the (subject, body) digest from the latest stored artifacts."""
    import pandas as pd

    ranking = store.read_parquet(Paths.LATEST_RANKING).sort_values(Cols.RANK)
    as_of = pd.Timestamp(ranking[Cols.DATE].max())
    buys = ranking[ranking[Cols.RATING].isin(["Strong Buy", "Buy"])]

    lines = [
        f"ETF Intel — weekly digest ({as_of:%Y-%m-%d})",
        "",
        "Automated ETF ranking. Research signals only — not financial advice.",
        "",
        "== Buy-side (Strong Buy / Buy) ==",
    ]
    for _, r in buys.iterrows():
        lines.append(f"  #{int(r[Cols.RANK])} {r[Cols.TICKER]} — {r[Cols.RATING]}")

    if store.exists(Paths.METRICS):
        m = store.read_json(Paths.METRICS)
        s, b = m.get("strategy", {}), m.get("benchmark", {})
        lines += [
            "",
            "== Walk-forward backtest (costs included) ==",
            f"  Strategy: CAGR {s.get('cagr', float('nan')) * 100:.1f}%  "
            f"Sharpe {s.get('sharpe', float('nan')):.2f}  "
            f"MaxDD {s.get('max_dd', float('nan')) * 100:.1f}%",
            f"  SPY:      CAGR {b.get('cagr', float('nan')) * 100:.1f}%  "
            f"Sharpe {b.get('sharpe', float('nan')):.2f}",
            f"  Skill (rank-IC): {m.get('skill', {}).get('mean_xs_rank_ic', float('nan')):+.3f}",
        ]
    return f"ETF Intel — weekly digest ({as_of:%Y-%m-%d})", "\n".join(lines)


def main() -> None:
    """Send the weekly digest to every configured channel."""
    _config, settings, _universe, store = load_context()
    if not store.exists(Paths.LATEST_RANKING):
        logger.warning("No ranking found. Run the pipeline first.")
        return

    subject, body = _build_body(store)
    sent = False
    if settings.resend_api_key and settings.report_recipient:
        ResendEmailAlertChannel(settings.resend_api_key, settings.report_recipient).send(
            subject, body
        )
        sent = True
    if settings.telegram_bot_token and settings.telegram_chat_id:
        TelegramAlertChannel(settings.telegram_bot_token, settings.telegram_chat_id).send(
            subject, body
        )
        sent = True
    if not sent:
        logger.info("No email/Telegram configured; digest not sent:\n%s", body)


if __name__ == "__main__":
    main()
