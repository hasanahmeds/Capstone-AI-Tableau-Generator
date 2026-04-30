"""
logger_config.py
----------------
Centralized Loguru setup for the project.

Every module calls get_logger("module_name") instead of using
stdlib logging.  Logs land in two places:
    1. stderr   (colored, for the terminal)
    2. logs/<module_name>.log  (rotated daily, kept 7 days)

Usage:
    from scripts.logger_config import get_logger
    logger = get_logger("my_module")
    logger.info("something happened")
"""

import os
import sys
from loguru import logger as _base_logger

# figure out where the project root is (one level up from scripts/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# strip loguru's default stderr sink so we control the format
_base_logger.remove()

# stderr sink — colored, human-readable
_base_logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{extra[module_name]}</cyan> — <level>{message}</level>",
    level="DEBUG",
    filter=lambda record: "module_name" in record["extra"],
)

# keep track of which modules already have a file sink
_registered_sinks = set()


def get_logger(module_name: str):
    """Return a loguru logger bound to *module_name*.

    The first call for a given name adds a per-module rotating
    file sink (logs/<module_name>.log).  Subsequent calls just
    return the bound logger — no duplicate sinks.
    """
    if module_name not in _registered_sinks:
        log_path = os.path.join(LOGS_DIR, f"{module_name}.log")
        _base_logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{extra[module_name]} — {message}",
            level="DEBUG",
            rotation="1 day",
            retention="7 days",
            filter=lambda record, mn=module_name: record["extra"].get("module_name") == mn,
            enqueue=True,
        )
        _registered_sinks.add(module_name)

    return _base_logger.bind(module_name=module_name)
