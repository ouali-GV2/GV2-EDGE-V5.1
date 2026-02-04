import logging
from logging.handlers import RotatingFileHandler
import os

from config import LOG_LEVEL, LOG_ROTATION_MB, LOG_BACKUPS

LOG_DIR = "data/logs"
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str):

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # avoid duplicate handlers

    logger.setLevel(getattr(logging, LOG_LEVEL))

    log_file = os.path.join(LOG_DIR, f"{name.lower()}.log")

    handler = RotatingFileHandler(
        log_file,
        maxBytes=LOG_ROTATION_MB * 1024 * 1024,
        backupCount=LOG_BACKUPS
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger
