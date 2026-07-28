"""Matplotlib plotting utilities (headless Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from etf_intel.common.types import Cols  # noqa: E402


def save_equity_curve(equity: pd.DataFrame, path: str | Path) -> Path:
    """Render strategy vs benchmark equity curves to a PNG.

    Args:
        equity: Frame with ``date``, ``strategy_equity``, ``benchmark_equity``.
        path: Output PNG path.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    if not equity.empty:
        ax.plot(equity[Cols.DATE], equity["strategy_equity"], label="Top-bucket strategy")
        ax.plot(
            equity[Cols.DATE],
            equity["benchmark_equity"],
            label="Benchmark",
            linestyle="--",
        )
    ax.set_title("Walk-forward equity curve (growth of 1)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
