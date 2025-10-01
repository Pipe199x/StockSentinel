import logging
import os

_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_logger = None

def get_logger(name: str) -> logging.Logger:
    global _logger
    if _logger is None:
        logging.basicConfig(level=_LEVEL, format=_FORMAT, datefmt=_DATEFMT)
    return logging.getLogger(name)
