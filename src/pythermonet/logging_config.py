from __future__ import annotations
import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    # DEV: logging.DEBUG / logging.INFO
    # PROD: logging.WARNING
    logger.setLevel(logging.INFO)
    return logger
