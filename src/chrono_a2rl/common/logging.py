"""Logging helpers."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "chrono_a2rl", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger without adding duplicate handlers."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    return logger
