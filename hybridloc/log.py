"""Simple timestamped logger — writes to stdout AND a rotating log file."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Log file path — override with HYBRIDLOC_LOG_FILE env var
_log_path: Path | None = None
_log_file = None


def _get_file():
    global _log_path, _log_file
    if _log_file is not None:
        return _log_file
    raw = os.environ.get("HYBRIDLOC_LOG_FILE", "logs/pipeline.log")
    _log_path = Path(raw)
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(_log_path, "a", buffering=1)  # line-buffered so tail -f works
    return _log_file


def log(msg: str, *, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    try:
        _get_file().write(line + "\n")
    except Exception:
        pass


def info(msg: str) -> None:
    log(msg, level="INFO")


def warn(msg: str) -> None:
    log(msg, level="WARN")


def error(msg: str) -> None:
    log(msg, level="ERROR")
