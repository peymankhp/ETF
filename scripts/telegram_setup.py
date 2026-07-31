"""Helper: discover your Telegram chat id for alerts.

Setup (2 minutes):
  1. In Telegram, message @BotFather -> /newbot -> copy the bot token.
  2. Put it in .env:  ETF_INTEL_TELEGRAM_BOT_TOKEN=123456:ABC...
  3. Open your new bot in Telegram and send it any message (e.g. "hi").
  4. Run this script -> it prints your chat id.
  5. Put it in .env:  ETF_INTEL_TELEGRAM_CHAT_ID=<the id>

Usage:
    python scripts/telegram_setup.py [--token <token>]
"""

from __future__ import annotations

import argparse

from _common import load_context
from etf_intel.common.logging import get_logger

logger = get_logger("telegram_setup")


def main() -> None:
    """Print chat ids from the bot's recent updates (send it a message first)."""
    import requests

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--token", default=None, help="bot token (defaults to .env)")
    args = ap.parse_args()

    _config, settings, _universe, _store = load_context()
    token = args.token or settings.telegram_bot_token
    if not token:
        logger.warning("No bot token. Set ETF_INTEL_TELEGRAM_BOT_TOKEN in .env or pass --token.")
        return

    resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=15)
    resp.raise_for_status()
    updates = resp.json().get("result", [])
    if not updates:
        logger.warning("No updates. Send your bot a message in Telegram first, then re-run.")
        return

    seen: set[str] = set()
    for upd in updates:
        chat = (upd.get("message") or upd.get("channel_post") or {}).get("chat") or {}
        cid = chat.get("id")
        if cid is not None and str(cid) not in seen:
            seen.add(str(cid))
            name = chat.get("title") or chat.get("username") or chat.get("first_name") or "?"
            logger.info("chat id: %s  (%s)", cid, name)
    logger.info("Add the id you want to .env as ETF_INTEL_TELEGRAM_CHAT_ID")


if __name__ == "__main__":
    main()
