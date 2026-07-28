"""Build point-in-time features + labels and train the LightGBM model.

Usage:
    python scripts/run_train.py
"""

from __future__ import annotations

from _common import load_context
from etf_intel.common.logging import get_logger
from etf_intel.datastore import Paths
from etf_intel.features import build_features
from etf_intel.features.pipeline import feature_columns
from etf_intel.labeling import TARGET_COL, build_labels
from etf_intel.models import assemble_training_frame, train_model
from etf_intel.models.registry import save_model

logger = get_logger("run_train")


def main() -> None:
    """Build features/labels, persist them, train the model and register it."""
    config, settings, universe, store = load_context()

    market = store.read_parquet(Paths.MARKET)
    macro = store.read_parquet(Paths.MACRO) if store.exists(Paths.MACRO) else None

    logger.info("Building features for %d rows of market data", len(market))
    features = build_features(market, macro, universe, config)
    labels = build_labels(market, universe, config)
    store.write_parquet(features, Paths.FEATURES)
    store.write_parquet(labels, Paths.LABELS)

    fcols = feature_columns(features)
    dataset = assemble_training_frame(features, labels)
    n_train = int(dataset[TARGET_COL].notna().sum())
    logger.info("Training on %d labelled rows, %d features", n_train, len(fcols))

    model = train_model(dataset, fcols, config)
    save_model(
        store,
        model,
        {
            "seed": config.seed,
            "n_features": len(fcols),
            "n_train_rows": n_train,
            "calibrated_classifier": model.classifier is not None,
        },
    )
    logger.info("Saved model -> %s (%d features)", Paths.MODEL, len(fcols))


if __name__ == "__main__":
    main()
