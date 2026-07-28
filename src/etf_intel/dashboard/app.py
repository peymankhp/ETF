"""Streamlit dashboard for ETF Intel.

Run with: ``streamlit run src/etf_intel/dashboard/app.py``.
Shows the latest ranking, a ticker detail view, the walk-forward equity curve,
and (best-effort) SHAP drivers for the selected ticker.
"""

from __future__ import annotations

# Import etf_intel first so lightgbm's OpenMP runtime loads before streamlit/pandas
# pull theirs in (Windows crash workaround; see etf_intel.__init__). Import order is
# deliberate here, so isort is disabled for this block.
# isort: off
import etf_intel  # noqa: F401
from etf_intel.common import load_config, load_settings
from etf_intel.common.types import Cols
from etf_intel.datastore import DataStore, Paths

import pandas as pd
import streamlit as st

# isort: on


def _store() -> DataStore:
    config = load_config()
    settings = load_settings()
    return DataStore(config.data_root(settings))


def _shap_drivers(store: DataStore, ticker: str) -> list[tuple[str, float]] | None:
    """Compute SHAP drivers for one ticker from the saved model + features."""
    try:
        from etf_intel.explain import ShapExplainer
        from etf_intel.models.registry import load_model

        features = store.read_parquet(Paths.FEATURES)
        latest_date = features[Cols.DATE].max()
        latest = features[features[Cols.DATE] == latest_date]
        row = latest[latest[Cols.TICKER] == ticker].set_index(Cols.TICKER)
        if row.empty:
            return None
        model = load_model(store)
        return ShapExplainer(model).top_drivers(row, top_n=6).get(ticker)
    except Exception as exc:  # pragma: no cover - dashboard best-effort
        st.info(f"SHAP unavailable: {exc}")
        return None


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="ETF Intel", layout="wide")
    st.title("ETF Intel — ETF ranking dashboard")
    st.caption("Research and educational signals. Not financial advice.")

    store = _store()
    if not store.exists(Paths.LATEST_RANKING):
        st.warning("No ranking found. Run the pipeline scripts first.")
        return

    ranking = store.read_parquet(Paths.LATEST_RANKING)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Latest ranking")
        st.dataframe(ranking, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Backtest metrics")
        if store.exists(Paths.METRICS):
            metrics = store.read_json(Paths.METRICS)
            strat = metrics.get("strategy", {})
            st.metric("Strategy CAGR", f"{strat.get('cagr', float('nan')) * 100:.2f}%")
            st.metric("Sharpe", f"{strat.get('sharpe', float('nan')):.2f}")
            st.metric("Max drawdown", f"{strat.get('max_dd', float('nan')) * 100:.2f}%")

    if store.exists(Paths.EQUITY_CURVE_PNG):
        st.subheader("Walk-forward equity curve")
        st.image(str(store.root / Paths.EQUITY_CURVE_PNG))

    st.subheader("Ticker detail")
    ticker = st.selectbox("Select a ticker", ranking[Cols.TICKER].tolist())
    if ticker:
        row = ranking[ranking[Cols.TICKER] == ticker].iloc[0]
        st.write(
            f"**{ticker}** — rating **{row[Cols.RATING]}**, "
            f"score {row[Cols.SCORE]:.4f}, "
            f"P(outperform) {row[Cols.PROB_OUTPERFORM] * 100:.1f}%"
        )
        if store.exists(Paths.FEATURES):
            features = store.read_parquet(Paths.FEATURES)
            hist = features[features[Cols.TICKER] == ticker]
            chart = hist.set_index(Cols.DATE)[Cols.ADJ_CLOSE]
            st.line_chart(chart)

        drivers = _shap_drivers(store, ticker)
        if drivers:
            st.write("**Top SHAP drivers:**")
            st.table(pd.DataFrame(drivers, columns=["feature", "shap_value"]))


if __name__ == "__main__":
    main()
