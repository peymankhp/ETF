"""Read-only FastAPI app serving stored rankings and backtest metrics.

The API never triggers trades or side effects — it only reads artefacts produced
by the pipeline scripts.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from etf_intel.common import load_config, load_settings
from etf_intel.common.types import Cols
from etf_intel.datastore import DataStore, Paths

app = FastAPI(
    title="ETF Intel API",
    description="Research signals for ETF ranking. Not financial advice.",
    version="0.1.0",
)


def _store() -> DataStore:
    config = load_config()
    settings = load_settings()
    return DataStore(config.data_root(settings))


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/ranking")
def ranking() -> list[dict[str, Any]]:
    """Return the latest cross-sectional ranking."""
    store = _store()
    if not store.exists(Paths.LATEST_RANKING):
        raise HTTPException(status_code=404, detail="No ranking available; run the pipeline.")
    df = store.read_parquet(Paths.LATEST_RANKING)
    return df.to_dict(orient="records")


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Return the latest backtest metrics."""
    store = _store()
    if not store.exists(Paths.METRICS):
        raise HTTPException(status_code=404, detail="No metrics available; run the backtest.")
    return store.read_json(Paths.METRICS)


@app.get("/ticker/{symbol}")
def ticker(symbol: str) -> dict[str, Any]:
    """Return the latest rating row for a single ticker."""
    store = _store()
    if not store.exists(Paths.LATEST_RANKING):
        raise HTTPException(status_code=404, detail="No ranking available; run the pipeline.")
    df = store.read_parquet(Paths.LATEST_RANKING)
    row = df[df[Cols.TICKER] == symbol.upper()]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Ticker {symbol!r} not found.")
    return row.iloc[0].to_dict()
