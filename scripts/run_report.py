"""Generate the weekly markdown ranking report from the trained model.

Usage:
    python scripts/run_report.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# NB: import _common (which loads etf_intel, preloading lightgbm's OpenMP on
# Windows) before importing pandas — see etf_intel.__init__. pandas is imported
# lazily inside main() for the same reason.
from _common import load_context
from etf_intel.common.logging import get_logger
from etf_intel.common.types import Cols
from etf_intel.datastore import Paths
from etf_intel.models.predict import predict
from etf_intel.models.registry import load_model
from etf_intel.models.train import TrainedModel
from etf_intel.portfolio.ranking import assign_ratings

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger("run_report")


def _explain(
    model: TrainedModel, latest: pd.DataFrame, tickers: list[str]
) -> dict[str, list[tuple[str, float]]] | None:
    """Best-effort SHAP drivers for the given tickers (None if SHAP unavailable)."""
    if not tickers:
        return None
    try:
        from etf_intel.explain import ShapExplainer

        rows = latest[latest[Cols.TICKER].isin(tickers)].set_index(Cols.TICKER)
        return ShapExplainer(model).top_drivers(rows, top_n=5)
    except Exception as exc:  # pragma: no cover - SHAP optional at report time
        logger.warning("SHAP explanation skipped: %s", exc)
        return None


def main() -> None:
    """Rank the latest date, explain the buys, and write the markdown report."""
    import pandas as pd  # imported after lightgbm preload (see module note)

    config, settings, universe, store = load_context()

    features = store.read_parquet(Paths.FEATURES)
    model = load_model(store)
    metrics = store.read_json(Paths.METRICS) if store.exists(Paths.METRICS) else {}

    latest_date = pd.Timestamp(features[Cols.DATE].max())
    latest = features[features[Cols.DATE] == latest_date].reset_index(drop=True)
    preds = predict(model, latest)
    latest[Cols.SCORE] = preds[Cols.SCORE].to_numpy()
    latest[Cols.PROB_OUTPERFORM] = preds[Cols.PROB_OUTPERFORM].to_numpy()

    ranking = assign_ratings(
        latest[[Cols.DATE, Cols.TICKER, Cols.SCORE, Cols.PROB_OUTPERFORM]], config.ratings
    )
    buys = ranking.loc[ranking[Cols.RATING].isin(["Strong Buy", "Buy"]), Cols.TICKER].tolist()
    explanations = _explain(model, latest, buys)

    from etf_intel.reporting import generate_markdown

    md = generate_markdown(latest_date, ranking, metrics, explanations)
    report_path = store.path(f"reports/weekly_{latest_date:%Y%m%d}.md")
    report_path.write_text(md, encoding="utf-8")
    store.write_parquet(ranking, Paths.LATEST_RANKING)

    logger.info("Report written -> %s", report_path)
    logger.info("Top buys: %s", ", ".join(buys) or "none")


if __name__ == "__main__":
    main()
