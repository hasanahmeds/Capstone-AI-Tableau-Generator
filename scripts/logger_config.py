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

# This file is in scripts folder
# — keeps log files out of the source tree
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)  # create logs/ on first run if it doesn't exist

# Loguru ships with a default stderr sink; we remove it so we can
# attach our own with a consistent format across the whole project
_base_logger.remove()

# stderr sink — colored output for whoever's watching the terminal
_base_logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{extra[module_name]}</cyan> — <level>{message}</level>",
    level="DEBUG",
    # Only show records that were created via get_logger() — avoids noise from libraries
    filter=lambda record: "module_name" in record["extra"],
)

# Track which modules already have a file sink so we don't add duplicates
# on repeated calls to get_logger() within the same process
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
            rotation="1 day",    # new file each day
            retention="7 days",  # automatically clean up files older than a week
            # Each file sink only captures records from its own module —
            # mn=module_name is a default arg trick to capture the loop variable correctly
            filter=lambda record, mn=module_name: record["extra"].get("module_name") == mn,
            enqueue=True,  # write from a background thread to avoid blocking the main process
        )
        _registered_sinks.add(module_name)

    return _base_logger.bind(module_name=module_name)
