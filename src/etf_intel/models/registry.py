"""On-disk model registry (pickle + JSON metadata) via the datastore."""

from __future__ import annotations

import pickle
from typing import Any

from etf_intel.datastore import DataStore, Paths
from etf_intel.models.train import TrainedModel


def save_model(store: DataStore, model: TrainedModel, meta: dict[str, Any]) -> None:
    """Persist a trained model and its metadata.

    Args:
        store: Target datastore.
        model: The fitted model to save.
        meta: Arbitrary metadata (seed, config hash, row counts, ...).
    """
    with store.path(Paths.MODEL).open("wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    store.write_json({"feature_cols": model.feature_cols}, Paths.FEATURE_COLS)
    store.write_json(
        {"target_col": model.target_col, "trained_at": model.trained_at, **meta},
        Paths.MODEL_META,
    )


def load_model(store: DataStore) -> TrainedModel:
    """Load a previously saved model.

    Args:
        store: Source datastore.

    Returns:
        The unpickled :class:`TrainedModel`.
    """
    with (store.root / Paths.MODEL).open("rb") as fh:
        model: TrainedModel = pickle.load(fh)
    return model
