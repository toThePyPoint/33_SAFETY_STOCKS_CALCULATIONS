import logging
from logging.handlers import RotatingFileHandler
import sys
import os
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _detect_project_root() -> Path:
    start = Path.cwd().resolve()

    def looks_like_project(p: Path) -> bool:
        return (p / "core").is_dir() and (p / "jobs").is_dir()

    for p in (start, *start.parents):
        if looks_like_project(p):
            return p

    return start


def _resolve_logfile_path(logfile: str) -> Path:
    lf = Path(logfile)

    if lf.is_absolute():
        return lf

    project_root = _detect_project_root()
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    os.environ.setdefault("RFC_TRACE_DIR", str(logs_dir))

    return logs_dir / lf.name


def setup_logger(name: str, logfile: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    log_path = _resolve_logfile_path(logfile)

    fh = RotatingFileHandler(
        log_path,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
