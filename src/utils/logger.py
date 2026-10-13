"""
Structured Logger
==================
Consistent logging across the entire project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "mtan_slr",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create and configure a logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level.
    log_file : str, optional
        If given, also writes logs to this file.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


get_logger = setup_logger
