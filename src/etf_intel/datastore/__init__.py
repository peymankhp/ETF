"""Persistence layer: versioned parquet snapshots + DuckDB access.

This is the *only* module that reads or writes disk. Every other module passes
in-memory frames/objects, which keeps the domain layer pure and testable.
"""

from etf_intel.datastore.store import DataStore, Paths

__all__ = ["DataStore", "Paths"]
