"""DuckDB + parquet datastore with versioned, hash-stamped snapshots.

Snapshots are written to a canonical path *and* an immutable timestamped copy,
each accompanied by a metadata sidecar (source, date range, ingested-at, row
count, content hash) so a run can be audited and reproduced.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


class Paths:
    """Canonical relative paths within the data root."""

    MARKET = "raw/market.parquet"
    MACRO = "raw/macro.parquet"
    FEATURES = "features/features.parquet"
    LABELS = "features/labels.parquet"
    MODEL = "models/model.pkl"
    MODEL_META = "models/model_meta.json"
    FEATURE_COLS = "models/feature_cols.json"
    PREDICTIONS = "backtest/predictions.parquet"
    METRICS = "backtest/metrics.json"
    EQUITY_CURVE = "backtest/equity_curve.parquet"
    EQUITY_CURVE_PNG = "backtest/equity_curve.png"
    LATEST_RANKING = "reports/latest_ranking.parquet"


def _content_hash(df: pd.DataFrame) -> str:
    """Return a stable SHA-256 over the frame's content (for reproducibility audits)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


class DataStore:
    """Read/write access to parquet snapshots and a DuckDB database."""

    def __init__(self, root: str | Path):
        """Initialise the store.

        Args:
            root: Data root directory (created if missing).
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- path helpers ------------------------------------------------------
    def path(self, rel: str) -> Path:
        """Return the absolute path for a store-relative path, creating parents."""
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self, rel: str) -> bool:
        """Return whether a store-relative path exists."""
        return (self.root / rel).exists()

    # -- parquet -----------------------------------------------------------
    def write_parquet(self, df: pd.DataFrame, rel: str) -> Path:
        """Write a frame to parquet (no index).

        Args:
            df: Frame to write.
            rel: Store-relative destination path.

        Returns:
            The absolute path written.
        """
        dest = self.path(rel)
        df.to_parquet(dest, engine="pyarrow", index=False)
        return dest

    def read_parquet(self, rel: str) -> pd.DataFrame:
        """Read a parquet file into a frame.

        Args:
            rel: Store-relative source path.

        Returns:
            The loaded frame.
        """
        return pd.read_parquet(self.root / rel, engine="pyarrow")

    # -- json --------------------------------------------------------------
    def write_json(self, obj: dict[str, Any], rel: str) -> Path:
        """Write a JSON-serialisable dict to ``rel``."""
        dest = self.path(rel)
        dest.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return dest

    def read_json(self, rel: str) -> dict[str, Any]:
        """Read a JSON file from ``rel`` into a dict."""
        return json.loads((self.root / rel).read_text(encoding="utf-8"))

    # -- versioned snapshots ----------------------------------------------
    def write_snapshot(
        self,
        df: pd.DataFrame,
        rel: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a snapshot plus an immutable timestamped copy and a metadata sidecar.

        Args:
            df: Snapshot frame.
            rel: Canonical store-relative parquet path (e.g. ``raw/market.parquet``).
            meta: Extra metadata to record (source, date range, ...).

        Returns:
            The full metadata dict that was written alongside the snapshot.
        """
        self.write_parquet(df, rel)
        stem = rel.rsplit(".", 1)[0]
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        versioned = f"{stem}.{stamp}.parquet"
        self.write_parquet(df, versioned)

        full_meta: dict[str, Any] = {
            "path": rel,
            "versioned_path": versioned,
            "ingested_at": datetime.now(UTC).isoformat(),
            "n_rows": int(len(df)),
            "content_sha256": _content_hash(df),
            **(meta or {}),
        }
        self.write_json(full_meta, f"{stem}.meta.json")
        return full_meta

    # -- duckdb ------------------------------------------------------------
    def connect(self, duckdb_file: str = "etf_intel.duckdb") -> Any:
        """Open (creating if needed) the DuckDB database in the data root.

        Args:
            duckdb_file: DuckDB filename relative to the data root.

        Returns:
            A live DuckDB connection.
        """
        import duckdb

        return duckdb.connect(str(self.root / duckdb_file))
