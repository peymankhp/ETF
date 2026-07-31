/**
 * On-demand Telegram bot on Cloudflare Workers (free, no credit card, always-on).
 *
 * Telegram calls this Worker (webhook) whenever you message the bot; it reads each
 * project's latest summary JSON from a public gist and replies. No server to keep
 * alive, nothing to maintain.
 *
 * Environment variables (Worker → Settings → Variables):
 *   BOT_TOKEN         Telegram bot token (from @BotFather)   [encrypt this one]
 *   ETF_SUMMARY_URL   raw gist URL of the ETF summary.json
 *   SPAI_SUMMARY_URL  raw gist URL of the SPAI summary.json  (optional)
 *
 * After deploy, point Telegram at it once (replace <TOKEN> and <WORKER_URL>):
 *   https://api.telegram.org/bot<TOKEN>/setWebhook?url=<WORKER_URL>&drop_pending_updates=true
 */

const DISCLAIMER = "Research signals only — not financial advice.";

async function fetchJson(url) {
  if (!url) return null;
  try {
    const r = await fetch(url, { cf: { cacheTtl: 60 } });
    if (!r.ok) return null;
    return await r.json();
  } catch (_) {
    return null;
  }
}

function pct(v) {
  return v === undefined || v === null ? "n/a" : (v * 100).toFixed(1) + "%";
}
function num(v) {
  return v === undefined || v === null ? "n/a" : Number(v).toFixed(2);
}
function money(v) {
  return v === undefined || v === null ? "n/a" : "$" + Math.round(v).toLocaleString("en-US");
}

function formatEtf(s) {
  if (!s) return "📊 *ETF Intel* — no data yet (waiting for the first weekly run).";
  const lines = [`📊 *ETF Intel* — ${s.as_of || "?"}`, "", "Buy-side:"];
  for (const b of (s.buys || []).slice(0, 12)) {
    lines.push(`  #${b.rank} ${b.ticker} — ${b.rating}`);
  }
  const bt = s.backtest;
  if (bt) {
    lines.push(
      "",
      `Backtest: CAGR ${pct(bt.cagr)} | Sharpe ${num(bt.sharpe)} | ` +
        `MaxDD ${pct(bt.max_dd)} | rank-IC ${bt.rank_ic == null ? "n/a" : Number(bt.rank_ic).toFixed(3)}`
    );
  }
  return lines.join("\n");
}

function formatSpai(s) {
  if (!s) return "🥇 *SPAI Gold* — no data yet.";
  return (
    `🥇 *SPAI Gold* — ${s.as_of || "?"}\n` +
    `  Price: ${money(s.gold_price)}\n` +
    `  Signal: ${s.direction || "?"} (${s.confidence_pct == null ? "?" : Math.round(s.confidence_pct)}% conf)\n` +
    `  7D target: ${money(s.target_7d)} | 30D: ${money(s.target_30d)}`
  );
}

async function reply(env, chatId, text) {
  await fetch(`https://api.telegram.org/bot${env.BOT_TOKEN}/sendMessage`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ chat_id: chatId, text, parse_mode: "Markdown" }),
  });
}

export default {
  async fetch(request, env) {
    if (request.method !== "POST") return new Response("bot ok"); // health check
    let update;
    try {
      update = await request.json();
    } catch (_) {
      return new Response("ok");
    }
    const msg = update.message || update.edited_message;
    if (!msg || !msg.text || !msg.chat) return new Response("ok");

    const chatId = msg.chat.id;
    const cmd = msg.text.trim().toLowerCase().split("@")[0];
    let body;
    if (cmd.startsWith("/etf")) {
      body = formatEtf(await fetchJson(env.ETF_SUMMARY_URL)) + `\n\n_${DISCLAIMER}_`;
    } else if (cmd.startsWith("/spai")) {
      body = formatSpai(await fetchJson(env.SPAI_SUMMARY_URL)) + `\n\n_${DISCLAIMER}_`;
    } else if (cmd.startsWith("/start") || cmd.startsWith("/help")) {
      const [etf, spai] = await Promise.all([
        fetchJson(env.ETF_SUMMARY_URL),
        fetchJson(env.SPAI_SUMMARY_URL),
      ]);
      body = [
        "👋 *Signals bot* — latest results on demand.",
        formatEtf(etf),
        formatSpai(spai),
        "Commands: /etf  /spai  /help",
        `_${DISCLAIMER}_`,
      ].join("\n\n");
    } else {
      body = "Commands: /start  /etf  /spai  /help";
    }
    await reply(env, chatId, body);
    return new Response("ok");
  },
};

// Exported for testing (harmless in the Worker runtime).
export { formatEtf, formatSpai };
