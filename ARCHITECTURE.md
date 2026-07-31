# ETF Intel — Architecture

## Design goals

- **Point-in-time correct** by construction: features are pure functions of the past;
  the only sanctioned use of future data is quarantined in `labeling/`.
- **Reproducible**: deterministic given `(config, seed, frozen data snapshot)`.
- **Modular (DDD-lite)**: clear bounded modules, a pure domain core, one-way dependencies,
  no cycles. Enforced by `import-linter` in CI.
- **Honest & measurable**: every experiment is MLflow-tracked; the read-out is out-of-sample
  skill (IC/rank-IC) plus cost-aware, stress-tested performance — not a single lucky number.
- **Solo-maintainable & automated**: plain CLI scripts, a one-command pipeline, and local
  weekly scheduling with email alerts. No orchestration server.

## Module boundaries & dependency direction

Dependencies point strictly downward. `import-linter` fails CI on any violation.

```
reporting | api | dashboard | tracking | alerting   (top: read-only consumers of stored outputs)
        │
     explain                     (SHAP over a trained model)
        │
     backtest                    (walk-forward integration layer + stress tests)
        │
 portfolio | models              (rank→buckets, weighting ; train/predict/ensemble/metrics)
        │
 labeling | features             (forward returns ; causal features — siblings, isolated)
        │
    datastore                    (the ONLY module that touches disk / DuckDB)
        │
    ingestion                    (provider interfaces + yfinance / FRED / synthetic / news)
        │
     common                      (config, logging, typed records, PIT price primitives)
```

Two hard rules beyond the layering:

- `etf_intel.features` **must never import** `etf_intel.labeling` (forbidden contract).
  Features cannot see labels, so they cannot accidentally leak the target.
- Only `datastore` performs I/O against parquet/DuckDB; every other module receives and
  returns in-memory frames/objects.

### Module responsibilities

| Module | Responsibility | May depend on |
|---|---|---|
| `common` | config models (pydantic-settings + YAML), logging, typed records, column names, the point-in-time total-return-index primitive | — |
| `ingestion` | `MarketDataProvider` / `MacroDataProvider` ABCs + yfinance / FRED / synthetic adapters (raw records); news `SentimentScorer` (lexicon / FinBERT) + yfinance headline provider | common |
| `datastore` | versioned, hash-stamped parquet snapshots + DuckDB; as-of reads | common |
| `features` | causal technical, cross-sectional, macro (release-lagged, off by default) and PIT trailing-dividend-yield features; deterministic split/div adjustment | common, datastore |
| `labeling` | forward returns (1w/1m/3m) and excess-vs-benchmark target — the one place future bars are used | common, datastore |
| `models` | LightGBM / XGBoost / averaging **ensemble**, calibrated classifier, metrics (IC/rank-IC/AUC), on-disk registry | common |
| `portfolio` | cross-sectional rank → 6 rating buckets; risk-aware **weighting** (equal / inverse-vol) with caps | common |
| `backtest` | walk-forward engine (purge + embargo, configurable rebalance, no-trade bands, transaction costs); performance + skill metrics; **stress tests** | common, features, labeling, models, portfolio |
| `explain` | SHAP values → top drivers per ETF (unwraps the ensemble) | common, models |
| `reporting` | weekly markdown report (ranking, metrics, skill, provenance) + equity-curve plot | downward only |
| `tracking` | log each backtest run (params, metrics, artifacts) to a local MLflow store | common |
| `alerting` | diff week-over-week ranking → alert via log / file / Resend email channels | common |
| `api` | FastAPI service exposing latest ranking + metrics | downward only |
| `dashboard` | Streamlit UI: ranking, ticker detail, equity curve, SHAP | downward only |

## Data flow

```
FRED ─┐                              ┌─ macro (release-lagged, off for x-sectional ranking) ─┐
      │  ingestion adapters          │                                                       │
yfinance ─► RAW unadjusted OHLCV + splits/divs ─► datastore (versioned parquet + duckdb)
                                     │                                                       │
                                     ├─ features/ PIT total-return index + causal indicators ┤
                                     │   + trailing dividend yield (as-of T)          feature matrix X(T)
labeling/ ── forward excess-vs-SPY (3m target) ──────────────────────────────────────┤  (quarantined)
                                                                                       ▼
              backtest/ walk-forward per rebalance date T (purge + embargo):
                 train LightGBM+XGBoost ensemble on rows ≤ cutoff → score at T
                 portfolio/ rank → 6 buckets → no-trade-band selection → weighting → costs
                                                                                       │
   ┌───────────────┬──────────────────────┬───────────────────┬──────────────────────┤
   ▼               ▼                      ▼                   ▼                      ▼
skill (IC /   equity curve +       stress/ crisis      explain/ SHAP        tracking/ MLflow
rank-IC/AUC)  perf metrics         windows + sub-       top drivers         (params+metrics+
                                   period stability                          artifacts)
   └───────────────┴──────────► reporting/ weekly markdown ◄────────────────┘
                                          │
        dashboard/ + api/ + alerting/ (ranking-change → email) + live news-sentiment overlay
```

## Point-in-time & anti-leakage safeguards

1. **Frozen raw snapshots (Decision C).** yfinance returns *retroactively* adjusted prices.
   We snapshot raw unadjusted OHLCV plus split/dividend series, and build a **point-in-time
   total-return index** from daily returns (no future-dividend back-adjustment), so a value
   at `T` depends only on data available at `T`.
2. **Causal features.** Every indicator (returns, vol, downside-vol, RSI, MACD, MA-distance,
   drawdown, trailing yield) uses trailing windows only. The leakage test perturbs *future*
   prices and asserts features at `T` are byte-identical; a positive control proves labels do
   change.
3. **Release-lagged macro (Decision B).** Macro is lagged by a conservative publication delay.
   (It is also *excluded from cross-sectional ranking* — it is constant across tickers on a
   date, so it carries no ranking signal.)
4. **Walk-forward with purge + embargo (Decision D).** Training uses only rows whose entire
   forward-label window closes before the fold cutoff, minus an embargo ≥ the label horizon.
5. **Prediction horizon ≠ holding period.** The model predicts the 3-month excess target, but
   the realised P&L per rebalance is the actual holding-period return — kept separate so the
   equity curve stays correct.
6. **Cross-sectional ranking within a date.** Buckets are assigned per rebalance date only.
7. **Sentiment is a live overlay, never in the backtest** — free historical news doesn't exist,
   so news sentiment augments the current ranking only and is deliberately kept out of the model.

## Reproducibility & tracking

- All parameters in `config/settings.yaml`; universe in `config/universe_v1.yaml` (~115 ETFs).
- Single `seed` threaded into the models. Committed `uv.lock`. Snapshots carry a metadata
  sidecar (source, date range, ingested-at, content hash).
- Every backtest run is logged to a local **MLflow** file store (`data/mlruns`).

## Current best configuration (on `master`)

114 ETFs · 3-month excess target · macro excluded from ranking · causal technicals + PIT
trailing yield · LightGBM+XGBoost ensemble · monthly rebalance, equal-weight with no-trade
bands (10%/40%) · 10 bps costs. Out-of-sample **rank-IC ≈ 0.11**, **CAGR ≈ 17% vs SPY ~16%**,
**Sharpe ≈ 1.03**, **MaxDD ≈ −18.6%** (better than SPY); edge persistent across time-thirds.

## Phased roadmap (status)

- **MVP — done.** 20 ETFs, LightGBM, walk-forward, SHAP, weekly report, Streamlit + FastAPI,
  Docker, tests, CI.
- **v1 — done.** ~115-ETF universe; XGBoost + ensembling; PIT trailing-dividend-yield feature;
  risk-aware weighting with caps + transaction costs; no-trade bands. (Key findings: breadth
  alone, inverse-vol, quarterly rebalance, HRP, and extra correlated features each *did not*
  help — kept the simplest configuration that maximises risk-adjusted return.)
- **v2 — done (data-limited parts noted).** MLflow tracking; one-command orchestration +
  weekly scheduling; news-sentiment overlay (lexicon + **FinBERT**); ranking-change alerting
  with **email via Resend**. *Deferred: Prefect/Airflow (over-engineering — the scheduler
  suffices); PatchTST/TFT (heavy, low expected value on tabular cross-sectional data).*
- **v3 — started.** Stress tests (crisis windows + sub-period skill stability). *Remaining
  (needs paid data): survivorship-bias-free / vintaged universe — the main honest caveat on
  the absolute numbers; alt-data; Kubernetes/Prometheus; Monte Carlo; full audit trail.*
