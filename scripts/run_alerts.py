"""Emit alerts for ranking changes since the previous run.

Compares the latest ranking to the previously saved snapshot, reports new/dropped
buy-side names and rating moves (log + file by default), then updates the snapshot.
Add this after run_pipeline in your weekly schedule to get change alerts.

Usage:
    python scripts/run_alerts.py
"""

from __future__ import annotations

from _common import load_context
from etf_intel.alerting import (
    FileAlertChannel,
    LogAlertChannel,
    ResendEmailAlertChannel,
    TelegramAlertChannel,
    format_alert,
    ranking_changes,
)
from etf_intel.common.logging import get_logger
from etf_intel.common.types import Cols
from etf_intel.datastore import Paths

logger = get_logger("run_alerts")
PREVIOUS_RANKING = "reports/previous_ranking.parquet"


def main() -> None:
    """Detect ranking changes vs the saved snapshot and deliver an alert."""
    import pandas as pd

    _config, settings, _universe, store = load_context()
    if not store.exists(Paths.LATEST_RANKING):
        logger.warning("No ranking found. Run the pipeline first.")
        return

    current = store.read_parquet(Paths.LATEST_RANKING)
    as_of = pd.Timestamp(current[Cols.DATE].max()) if Cols.DATE in current else pd.Timestamp.today()

    if store.exists(PREVIOUS_RANKING):
        previous = store.read_parquet(PREVIOUS_RANKING)
        changes = ranking_changes(previous, current)
        body = format_alert(changes, as_of)
        LogAlertChannel().send("ETF Intel ranking update", body)
        if not changes.is_empty():
            subject = f"ETF Intel ranking update ({as_of:%Y-%m-%d})"
            FileAlertChannel(store.path("reports/alerts")).send(subject, body)
            # Email via Resend when configured.
            if settings.resend_api_key and settings.report_recipient:
                ResendEmailAlertChannel(settings.resend_api_key, settings.report_recipient).send(
                    subject, body
                )
            # Telegram push when configured.
            if settings.telegram_bot_token and settings.telegram_chat_id:
                TelegramAlertChannel(settings.telegram_bot_token, settings.telegram_chat_id).send(
                    subject, body
                )
    else:
        logger.info("No previous ranking snapshot; saving baseline (no alert this run).")

    store.write_parquet(current, PREVIOUS_RANKING)


if __name__ == "__main__":
    main()
