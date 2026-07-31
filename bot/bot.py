"""On-demand Telegram bot: replies with the latest ETF + SPAI results.

Host-agnostic long-polling bot (raw ``requests``, no heavy deps) — deploy it on
any always-on host (Fly.io, a free VM, a Raspberry Pi, ...). It reads compact JSON
summaries that each project's weekly GitHub Actions run publishes to a public gist,
so the bot itself needs no GitHub credentials.

Commands: /start, /etf, /spai, /help.

Environment variables:
    BOT_TOKEN         Telegram bot token (from @BotFather)
    ETF_SUMMARY_URL   raw URL to the ETF summary JSON (e.g. a gist raw link)
    SPAI_SUMMARY_URL  raw URL to the SPAI summary JSON (optional)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("etf_spai_bot")

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
ETF_SUMMARY_URL = os.environ.get("ETF_SUMMARY_URL", "")
SPAI_SUMMARY_URL = os.environ.get("SPAI_SUMMARY_URL", "")
API = f"https://api.telegram.org/bot{BOT_TOKEN}"
DISCLAIMER = "Research signals only — not financial advice."


def _fetch(url: str) -> dict[str, Any] | None:
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return dict(resp.json())
    except Exception as exc:  # network / parse errors -> None, handled by caller
        log.warning("fetch failed (%s): %s", url, exc)
        return None


def _format_etf(s: dict[str, Any] | None) -> str:
    if not s:
        return "📊 *ETF Intel* — no data yet (waiting for the first weekly run)."
    lines = [f"📊 *ETF Intel* — {s.get('as_of', '?')}", "", "Buy-side:"]
    for b in s.get("buys", [])[:12]:
        lines.append(f"  #{b.get('rank')} {b.get('ticker')} — {b.get('rating')}")
    bt = s.get("backtest", {})
    if bt:
        lines += [
            "",
            f"Backtest: CAGR {bt.get('cagr', 0) * 100:.1f}% | "
            f"Sharpe {bt.get('sharpe', float('nan')):.2f} | "
            f"MaxDD {bt.get('max_dd', 0) * 100:.1f}% | "
            f"rank-IC {bt.get('rank_ic', float('nan')):+.3f}",
        ]
    return "\n".join(lines)


def _format_spai(s: dict[str, Any] | None) -> str:
    if not s:
        return "🥇 *SPAI Gold* — no data yet."
    return (
        f"🥇 *SPAI Gold* — {s.get('as_of', '?')}\n"
        f"  Price: ${s.get('gold_price', 0):,.0f}\n"
        f"  Signal: {s.get('direction', '?')} ({s.get('confidence_pct', 0):.0f}% conf)\n"
        f"  7D target: ${s.get('target_7d', 0):,.0f} | 30D: ${s.get('target_30d', 0):,.0f}"
    )


def _send(chat_id: int, text: str) -> None:
    try:
        requests.post(
            f"{API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
    except Exception as exc:
        log.warning("send failed: %s", exc)


def handle(text: str, chat_id: int) -> None:
    """Route a command to a reply."""
    cmd = text.strip().lower().split("@")[0]
    if cmd.startswith("/etf"):
        _send(chat_id, _format_etf(_fetch(ETF_SUMMARY_URL)) + f"\n\n_{DISCLAIMER}_")
    elif cmd.startswith("/spai"):
        _send(chat_id, _format_spai(_fetch(SPAI_SUMMARY_URL)) + f"\n\n_{DISCLAIMER}_")
    elif cmd.startswith("/start") or cmd.startswith("/help"):
        body = "\n\n".join(
            [
                "👋 *Signals bot* — latest results on demand.",
                _format_etf(_fetch(ETF_SUMMARY_URL)),
                _format_spai(_fetch(SPAI_SUMMARY_URL)),
                "Commands: /etf  /spai  /help",
                f"_{DISCLAIMER}_",
            ]
        )
        _send(chat_id, body)
    else:
        _send(chat_id, "Commands: /start  /etf  /spai  /help")


def main() -> None:
    """Long-poll Telegram and reply to commands."""
    if not BOT_TOKEN:
        raise SystemExit("BOT_TOKEN not set")
    log.info("bot started (long-polling)")
    offset: int | None = None
    while True:
        try:
            resp = requests.get(
                f"{API}/getUpdates",
                params={"timeout": 30, "offset": offset},
                timeout=40,
            )
            for upd in resp.json().get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message") or {}
                text = msg.get("text", "")
                chat = (msg.get("chat") or {}).get("id")
                if text and chat is not None:
                    log.info("msg from %s: %s", chat, text)
                    handle(text, int(chat))
        except Exception as exc:  # keep the loop alive on transient errors
            log.warning("poll error: %s", exc)


if __name__ == "__main__":
    main()
