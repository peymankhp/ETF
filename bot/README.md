# Signals Bot (@SPAI_ETF_bot) — on-demand ETF + SPAI results

A tiny always-on Telegram bot. Message it `/start` (or `/etf`, `/spai`) and it
replies with each project's **latest** results. It reads compact JSON summaries
that each project's weekly GitHub Actions run publishes to a public **gist**, so
the bot needs no GitHub credentials and stays trivially cheap to host.

```
you → /start → bot → reads ETF gist + SPAI gist → replies with both
```

## Why it needs hosting
Telegram must reach the bot 24/7, so it can't live on your (off) laptop or in
GitHub Actions (which only runs on a schedule). It runs as one small always-on
process. Any of these work: **Fly.io** (free allowance), a **free VM** (Oracle
Cloud Always-Free), a **Raspberry Pi**, etc.

## Setup (one time)

### 1. Publish each project's summary to a gist
Each weekly run writes a small `summary.json`; a workflow step PATCHes it into a
gist. You need:
- **A GitHub token with the `gist` scope** — github.com → Settings → Developer
  settings → Personal access tokens → *Fine-grained or classic* → enable **gist**.
- **Two secret gists** (gist.github.com → create → any content → Create secret
  gist). Copy each gist **ID** (the hash in its URL) and its **raw** file URL.
- Add these as **repo secrets** in *both* ETF and SPAI:
  `GIST_TOKEN` (the token), `GIST_ID` (that project's gist id).

The ETF workflow already includes the publish step (`build_summary.py` + a
`curl` PATCH). SPAI: add the same pattern (see the ETF workflow for reference).

### 2. Deploy the bot (Fly.io example)
```bash
# one-time: install flyctl and sign in (free account)
#   https://fly.io/docs/hands-on/install-flyctl/   then:  fly auth signup

cd bot
fly launch --no-deploy --copy-config --name etf-spai-bot
fly secrets set \
  BOT_TOKEN="8858366021:AA...your-bot-token" \
  ETF_SUMMARY_URL="https://gist.githubusercontent.com/<you>/<etf_gist>/raw/summary.json" \
  SPAI_SUMMARY_URL="https://gist.githubusercontent.com/<you>/<spai_gist>/raw/summary.json"
fly deploy
```
That's it — message the bot `/start`. (Any other host: run the Docker image, or
`pip install -r requirements.txt && python bot.py`, with the same three env vars.)

## Env vars
| Var | Meaning |
|---|---|
| `BOT_TOKEN` | Telegram bot token (@BotFather) |
| `ETF_SUMMARY_URL` | raw gist URL of the ETF `summary.json` |
| `SPAI_SUMMARY_URL` | raw gist URL of the SPAI `summary.json` (optional) |

## Local test
```bash
BOT_TOKEN=... ETF_SUMMARY_URL=... python bot.py
```
