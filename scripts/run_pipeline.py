"""Run the whole pipeline end-to-end in one command.

ingest -> train -> backtest -> report. Any extra args (e.g. ``--source synthetic``)
are forwarded to the ingest step only.

Usage:
    python scripts/run_pipeline.py [--source synthetic|live]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def main() -> None:
    """Run ingest, train, backtest and report in sequence."""
    ingest_args = sys.argv[1:]  # forwarded to ingest only (it owns --source)
    steps: list[list[str]] = [
        [sys.executable, str(SCRIPTS / "run_ingest.py"), *ingest_args],
        [sys.executable, str(SCRIPTS / "run_train.py")],
        [sys.executable, str(SCRIPTS / "run_backtest.py")],
        [sys.executable, str(SCRIPTS / "run_report.py")],
        [sys.executable, str(SCRIPTS / "run_stress.py")],
        [sys.executable, str(SCRIPTS / "run_alerts.py")],
    ]
    for cmd in steps:
        print(f"\n=== {Path(cmd[1]).name} ===", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
