# /start bot on Cloudflare Workers — free, no credit card, always-on

The bot as a Cloudflare Worker (webhook). Telegram calls it when you message the
bot; it reads each project's latest `summary.json` from a public gist and replies.
Nothing to keep running, no VM, no card. Cloudflare's free plan is plenty
(100k requests/day).

```
you → /start → Telegram → Cloudflare Worker → reads ETF gist + SPAI gist → replies
```

## Prerequisite: the two gists (data)
Do the gist setup first (see `../README.md` → "Publish each project's summary to a
gist"): create the two secret gists, add `GIST_TOKEN` + `GIST_ID` secrets to the
ETF and SPAI repos, and run each weekly workflow once so the gists get populated.
Note each gist's **raw** URL (e.g. `https://gist.githubusercontent.com/<you>/<id>/raw/summary.json`).

## Deploy — dashboard (no CLI, ~5 min)
1. Sign up free at **https://dash.cloudflare.com/sign-up** (no credit card for Workers).
2. **Workers & Pages → Create → Create Worker** → give it a name (e.g. `etf-spai-bot`) → **Deploy**.
3. **Edit code** → delete the sample → paste all of **`worker.js`** → **Deploy**.
4. **Settings → Variables and Secrets** → add:
   - `BOT_TOKEN` — your bot token (click **Encrypt**)
   - `ETF_SUMMARY_URL` — the ETF gist raw URL
   - `SPAI_SUMMARY_URL` — the SPAI gist raw URL
   → **Deploy** again to apply.
5. Copy your Worker URL (e.g. `https://etf-spai-bot.<you>.workers.dev`).
6. **Point Telegram at it** — open this once in a browser (replace both bits):
   ```
   https://api.telegram.org/bot<BOT_TOKEN>/setWebhook?url=<WORKER_URL>&drop_pending_updates=true
   ```
   You should see `{"ok":true,...}`.

Now message the bot **/start** → it replies with both projects. Done — free and
always on.

## Deploy — CLI alternative (wrangler)
```bash
cd bot/cloudflare
npx wrangler deploy
npx wrangler secret put BOT_TOKEN        # paste the token
# set ETF_SUMMARY_URL / SPAI_SUMMARY_URL in wrangler.toml [vars], redeploy
# then the setWebhook URL above
```

## Notes
- Setting a webhook disables long-polling — use the webhook Worker OR the Python
  polling bot (`../bot.py`), not both.
- To remove the webhook later: `https://api.telegram.org/bot<BOT_TOKEN>/deleteWebhook`.
