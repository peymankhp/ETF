"""Local MLflow experiment tracking for backtest runs.

Logs a flattened view of the config, the skill + performance metrics, and key
artifacts to a local file-backed MLflow store, so every experiment (all the config
comparisons we run) is reproducible and diff-able. Best-effort: a tracking failure
never breaks the pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from etf_intel.common.config import AppConfig
from etf_intel.common.logging import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME = "etf-intel-backtest"


def _flatten_params(config: AppConfig) -> dict[str, Any]:
    """Pull the decision-relevant config knobs into a flat param dict."""
    return {
        "seed": config.seed,
        "universe_file": config.universe_file,
        "target_horizon": config.target.horizon,
        "target_kind": config.target.kind,
        "model_kind": config.model.kind,
        "include_macro": config.features.include_macro,
        "rebalance": config.backtest.rebalance,
        "retrain_every_months": config.backtest.retrain_every_months,
        "portfolio_scheme": config.portfolio.scheme,
        "no_trade_bands": config.portfolio.no_trade_bands,
        "entry_top_frac": config.portfolio.entry_top_frac,
        "exit_top_frac": config.portfolio.exit_top_frac,
        "cost_bps": config.portfolio.cost_bps,
    }


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Flatten nested backtest metrics into ``mlflow.log_metrics``-friendly floats."""
    out: dict[str, float] = {}
    for key in ("n_periods", "avg_turnover", "excess_hit_rate"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(value)
    for group in ("strategy", "benchmark", "skill"):
        sub = metrics.get(group)
        if isinstance(sub, dict):
            for name, value in sub.items():
                if isinstance(value, (int, float)):
                    out[f"{group}_{name}"] = float(value)
    return out


def log_backtest_run(
    config: AppConfig,
    metrics: dict[str, Any],
    tracking_dir: str | Path,
    artifacts: Sequence[str | Path] | None = None,
    run_name: str | None = None,
) -> str | None:
    """Log one backtest run to the local MLflow store.

    Args:
        config: The application config used for the run.
        metrics: The metrics dict from ``compute_backtest``.
        tracking_dir: Directory for the MLflow file store (e.g. ``data/mlruns``).
        artifacts: Optional file paths to attach (equity curve, metrics json, ...).
        run_name: Optional human-readable run name.

    Returns:
        The MLflow run id, or ``None`` if tracking was skipped/failed.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(Path(tracking_dir).resolve().as_uri())
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(_flatten_params(config))
            mlflow.log_metrics(_flatten_metrics(metrics))
            for path in artifacts or []:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
        logger.info("Logged run %s to MLflow (%s)", run.info.run_id, EXPERIMENT_NAME)
        return str(run.info.run_id)
    except Exception as exc:  # pragma: no cover - tracking is best-effort
        logger.warning("MLflow logging skipped: %s", exc)
        return None
