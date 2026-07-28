# ETF Intel — Architecture

## Design goals

- **Point-in-time correct** by construction: features are pure functions of the past;
  the only sanctioned use of future data is quarantined in `labeling/`.
- **Reproducible**: deterministic given `(config, seed, frozen data snapshot)`.
- **Modular (DDD-lite)**: clear bounded modules, a pure domain core, one-way dependencies,
  no cycles. Enforced by `import-linter` in CI.
- **Solo-maintainable**: plain CLI scripts, no orchestration server in the MVP.

## Module boundaries & dependency direction

Dependencies point strictly downward. `import-linter` fails CI on any violation.

```
reporting | api | dashboard      (top: read-only consumers of stored outputs)
        │
     explain                     (SHAP over a trained model)
        │
     backtest                    (walk-forward integration layer)
        │
 portfolio | models              (ranking → buckets ; train/predict/metrics)
        │
 labeling | features             (forward returns ; causal features — siblings, isolated)
        │
    datastore                    (the ONLY module that touches disk / DuckDB)
        │
    ingestion                    (provider interfaces + yfinance / FRED / synthetic)
        │
     common                      (config, logging, typed records, as-of clock)
```

Two hard rules beyond the layering:

- `etf_intel.features` **must never import** `etf_intel.labeling` (forbidden contract).
  Features cannot see labels, so they cannot accidentally leak the target.
- Only `datastore` performs I/O against parquet/DuckDB; every other module receives and
  returns in-memory frames/objects.

### Module responsibilities

| Module | Responsibility | May depend on |
|---|---|---|
| `common` | config models (pydantic-settings + YAML), logging, typed records, column-name constants, `as_of` helpers | — |
| `ingestion` | `MarketDataProvider` / `MacroDataProvider` ABCs; concrete `yfinance`, `FRED`, `synthetic` adapters returning **raw** records | common |
| `datastore` | versioned parquet snapshots + DuckDB views; as-of reads | common |
| `features` | causal technical, cross-sectional, macro (release-lagged) features; deterministic split/dividend adjustment | common, datastore |
| `labeling` | forward returns (1w/1m/3m) and excess-vs-benchmark labels — the one place future bars are used | common, datastore |
| `models` | LightGBM train/predict/calibration, metrics, on-disk model registry | common |
| `portfolio` | cross-sectional rank → 6 rating buckets within each rebalance date | common |
| `backtest` | walk-forward engine (purge + embargo), performance metrics, equity curve | common, features, labeling, models, portfolio |
| `explain` | SHAP values → top drivers per ETF | common, models |
| `reporting` | weekly markdown report (ranking tables + metrics) | downward only |
| `api` | FastAPI service exposing latest ranking + explanation | downward only |
| `dashboard` | Streamlit UI: ranking, ticker detail, SHAP | downward only |

## Data flow

```
FRED ─┐                                   ┌─ macro features (release-lagged, Decision B) ─┐
      │  ingestion adapters (interface)   │                                               │
yfinance ─► RAW unadjusted OHLCV + splits/divs snapshot ─► datastore (versioned parquet/duckdb)
                                          │                                               │
                                          ├─ features/ deterministic adjustment + causal indicators (as-of T) ─┤
                                          │                                               │                    ▼
                                          │                                        feature matrix X(T)
labeling/ ── forward returns y(T→T+h), excess vs SPY  ────────────────────────────────────┤   (quarantined)
                                                                                           ▼
                       backtest/ walk-forward, per fold:
                          train on rows ≤ T_train with label window purged + embargo gap
                          predict at T_test → score + P(outperform SPY)
                                                                                           │
              ┌─────────────────────────────┬──────────────────────────┬──────────────────┤
              ▼                             ▼                          ▼                  ▼
   portfolio/ rank → 6 buckets   metrics (CAGR/Sharpe/Sortino/    explain/ SHAP      equity curve
   (Strong Buy … Strong Sell)    MaxDD/hit rate)                  top drivers
              │                             │                          │
              └──────────────► reporting/ weekly markdown ◄────────────┘
                                          │
                                dashboard/ (Streamlit) + api/ (FastAPI)
```

## Point-in-time & anti-leakage safeguards

1. **Frozen raw snapshots (Decision C).** yfinance returns *retroactively* adjusted prices.
   We snapshot raw unadjusted OHLCV plus explicit split/dividend series to versioned
   parquet, and compute adjustments deterministically in `features` from the frozen
   snapshot. This preserves "same inputs → same outputs".
2. **Causal features.** Every indicator uses trailing windows only (`.rolling(...)` /
   `.shift(1)` where a value would otherwise peek at `T`). The leakage test perturbs
   *future* prices and asserts features at `T` are byte-identical.
3. **Release-lagged macro (Decision B).** Each FRED series is lagged by a conservative
   publication delay so a feature at `T` never uses a value the market had not yet seen.
   True ALFRED vintages are a v1 upgrade.
4. **Walk-forward with purge + embargo (Decision D).** Training uses only rows whose entire
   forward-label window closes on or before the fold's cutoff, minus an embargo ≥ the label
   horizon. No label window straddles the train/test boundary.
5. **Cross-sectional ranking within a date.** Rating buckets are assigned per rebalance
   date, never pooled across time.

## Reproducibility

- All parameters in `config/settings.yaml`; universe in `config/universe.yaml`.
- Single `seed` threaded into the model and any sampling.
- Committed `uv.lock`; data snapshots carry a metadata sidecar (source, date range,
  ingested-at timestamp).

## Phased roadmap

- **MVP (this repo):** 20 ETFs, LightGBM baseline, walk-forward backtest, SHAP, weekly
  report, Streamlit + FastAPI, Docker, tests, CI.
- **v1:** few-hundred ETF universe; XGBoost/CatBoost + ensembling; free fundamentals
  (expense ratio, AUM, yield); PyPortfolioOpt allocation (HRP + risk parity) with
  transaction-cost & position limits; hardened FastAPI service.
- **v2:** news + LLM pipeline (FinBERT, entity/event extraction, embeddings + RAG);
  Prefect/Airflow orchestration; MLflow model registry; PatchTST/TFT vs the GBM baseline;
  email/Telegram alerting.
- **v3 (institutional):** paid providers behind existing interfaces; alt data; Kubernetes;
  Prometheus/Grafana; stress tests (2008/COVID/rate shocks); Monte Carlo & rolling-window
  validation; full audit trail.
