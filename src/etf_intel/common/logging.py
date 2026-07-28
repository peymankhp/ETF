"""Minimal, consistent logging setup for scripts and library code."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring root handlers once.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A configured :class:`logging.Logger`.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.environ.get("ETF_INTEL_LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
