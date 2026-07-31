# ETF Intel

A solo-maintainable, institutional-quality **AI ETF research platform**. It ingests ETF
price and macro data, engineers point-in-time features, trains a model to score expected
risk-adjusted outperformance, walk-forward backtests it, and ranks each ETF into six
buckets — **Strong Buy / Buy / Hold / Reduce / Sell / Strong Sell** — each with a
confidence score and a human-readable SHAP explanation.

> ⚠️ **This is a research and educational tool, not a licensed financial advisor.**
> Ratings are signals for a human to review, never automated orders.

**Status: MVP + v1 + v2 complete, v3 started.** ~115 ETFs, a LightGBM+XGBoost ensemble,
risk-aware portfolio construction with transaction costs, MLflow-tracked experiments,
FinBERT news-sentiment overlay, email alerts, and a stress-tested, point-in-time-correct
walk-forward backtest. Current best config beats SPY on return **and** drawdown (CAGR ~17%
vs ~16%, Sharpe ~1.03, MaxDD −18.6%) with an out-of-sample edge (rank-IC ~0.11) that is
persistent across time. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design, current
configuration, and the phased roadmap with status.

## Core principles

1. **Point-in-time correctness** — a feature for date `T` uses only information available
   at `T`. Labels are forward-looking; evaluation is walk-forward. A dedicated
   [leakage test](tests/test_leakage.py) fails if a feature ever consumes future data.
2. **Reproducibility** — fixed seeds, all config in YAML, data written to versioned
   parquet + DuckDB. Same inputs → same outputs.
3. **Clean modular architecture** — strict one-way dependencies between ingestion →
   datastore → features/labeling → models/portfolio → backtest → explain → reporting.
   Enforced in CI by [import-linter](pyproject.toml).

## Setup

Requires **Python 3.11** and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (once)
pip install uv                      # or: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the environment and install (dev extras include lint/type/test tools)
uv venv
uv sync --extra dev

# 3. Configure secrets
cp .env.example .env                # then paste your free FRED API key (optional)
```

Everything runs **offline** with synthetic data if you have no FRED key / no network —
set `ETF_INTEL_MARKET_SOURCE=synthetic` and `ETF_INTEL_MACRO_SOURCE=synthetic` in `.env`.
This is also how CI runs.

## Run the pipeline

Each step is one documented command. Run them in order from a clean checkout:

```bash
uv run python scripts/run_ingest.py     # ingest OHLCV (yfinance) + macro (FRED) -> data/
uv run python scripts/run_train.py      # build features + labels, train LightGBM -> data/models/
uv run python scripts/run_backtest.py   # walk-forward backtest -> metrics + equity curve
uv run python scripts/run_report.py     # weekly markdown ranking report -> data/reports/
```

Offline demo (no network/keys), useful for a first run and for CI:

```bash
uv run python scripts/run_ingest.py --source synthetic
uv run python scripts/run_train.py
uv run python scripts/run_backtest.py
uv run python scripts/run_report.py
```

### One command + weekly scheduling

Run the whole pipeline (ingest → train → backtest → report) in one step:

```bash
uv run python scripts/run_pipeline.py            # live data
uv run python scripts/run_pipeline.py --source synthetic
```

Every backtest is logged to a local **MLflow** store (`data/mlruns`) — params,
skill + performance metrics, and artifacts — so experiments are reproducible and
comparable. Browse them with:

```bash
uv run mlflow ui --backend-store-uri data/mlruns
```

**Schedule it weekly** (refreshes the ranking on its own):

- **Windows (Task Scheduler)** — run every Monday 07:00:
  ```bat
  schtasks /Create /TN "ETF-Intel weekly" /SC WEEKLY /D MON /ST 07:00 ^
    /TR "cmd /c cd /d D:\Github\ETF && uv run python scripts/run_pipeline.py"
  ```
- **Linux/macOS (cron)** — `crontab -e`, then:
  ```cron
  0 7 * * 1 cd /path/to/etf-intel && uv run python scripts/run_pipeline.py
  ```

### Live news sentiment (optional overlay)

A live sentiment overlay on the current top names (not part of the backtested
model — free historical news makes sentiment un-backtestable):

```bash
uv run python scripts/run_sentiment.py                     # fast lexicon scorer
uv sync --extra finbert                                    # one-time: torch + transformers
uv run python scripts/run_sentiment.py --model finbert     # FinBERT (finance-tuned)
```

### Dashboard & API

```bash
uv run streamlit run src/etf_intel/dashboard/app.py     # ranking + ticker detail + SHAP
uv run uvicorn etf_intel.api.app:app --reload           # FastAPI (http://127.0.0.1:8000/docs)
```

## Quality gates

```bash
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run mypy                  # static types
uv run lint-imports          # architecture contracts (import-linter)
uv run pytest                # tests incl. the leakage test
```

CI runs all of the above on every push (see `.github/workflows/ci.yml`).

## Docker

```bash
docker compose build
docker compose run --rm etf-intel python scripts/run_ingest.py --source synthetic
```

## Repository layout

```
config/          settings.yaml + universe_v1.yaml (~115 tickers)
src/etf_intel/   ingestion, datastore, features, labeling, models, backtest,
                 portfolio, explain, reporting, tracking, alerting, api,
                 dashboard, common
scripts/         run_ingest / run_train / run_backtest / run_report / run_stress /
                 run_alerts / run_sentiment / run_pipeline (one command)
tests/           unit tests + leakage test + architecture contract test
data/            (gitignored) versioned parquet snapshots, duckdb, mlruns
```
