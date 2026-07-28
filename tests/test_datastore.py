"""Tests for the parquet/JSON datastore and versioned snapshots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from etf_intel.datastore import DataStore


def test_parquet_roundtrip(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    store.write_parquet(df, "sub/dir/data.parquet")
    loaded = store.read_parquet("sub/dir/data.parquet")
    pd.testing.assert_frame_equal(df, loaded)


def test_snapshot_writes_meta_with_stable_hash(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3]})
    meta1 = store.write_snapshot(df, "raw/market.parquet", {"source": "synthetic"})
    meta2 = store.write_snapshot(df.copy(), "raw/market.parquet", {"source": "synthetic"})

    assert meta1["content_sha256"] == meta2["content_sha256"]  # deterministic
    assert meta1["n_rows"] == 3
    assert store.exists("raw/market.meta.json")
    assert store.read_json("raw/market.meta.json")["source"] == "synthetic"
