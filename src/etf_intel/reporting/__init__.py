"""Reporting: weekly markdown ranking report and equity-curve plotting."""

from etf_intel.reporting.plots import save_equity_curve
from etf_intel.reporting.weekly import generate_markdown

__all__ = ["generate_markdown", "save_equity_curve"]
