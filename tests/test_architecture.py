"""Architecture guardrails mirrored as a fast unit test.

The full contract set is enforced in CI via ``lint-imports``. Here we assert the
single most important anti-leakage rule directly, so it fails fast in ``pytest``
too: the ``features`` package must never import ``labeling``.
"""

from __future__ import annotations

import ast
from pathlib import Path

FEATURES_DIR = Path(__file__).resolve().parents[1] / "src" / "etf_intel" / "features"


def _imported_modules(source: str) -> set[str]:
    """Return the set of module names imported by a Python source string."""
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_features_never_import_labeling() -> None:
    offenders: list[str] = []
    for py in FEATURES_DIR.rglob("*.py"):
        modules = _imported_modules(py.read_text(encoding="utf-8"))
        if any(m == "etf_intel.labeling" or m.startswith("etf_intel.labeling.") for m in modules):
            offenders.append(py.name)
    assert not offenders, f"features must not import labeling (found in {offenders})"
