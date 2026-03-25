import logging
import os
import sys


def bootstrap_logging(level: str | None = None) -> None:
    """Configure process-wide logging once with a sensible default formatter."""
    level_name = (level or os.getenv("HEXAPOD_LOG_LEVEL") or "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
