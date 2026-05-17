"""Simple timestamped logger — writes to stdout, a global pipeline.log, and per-instance logs."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# Global log file (all instances combined)
_log_path: Path | None = None
_log_file = None

# Per-instance log file (set via set_instance / instance_log context manager)
_instance_file = None


def _get_file():
    global _log_path, _log_file
    if _log_file is not None:
        return _log_file
    raw = os.environ.get("HYBRIDLOC_LOG_FILE", "logs/pipeline.log")
    _log_path = Path(raw)
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(_log_path, "a", buffering=1)
    return _log_file


@contextmanager
def instance_log(instance_id: str, log_dir: str = "logs/instances"):
    """Context manager that opens a per-instance log file for the duration of one run."""
    global _instance_file
    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{instance_id}.log"
    prev = _instance_file
    _instance_file = open(path, "w", buffering=1)
    try:
        yield path
    finally:
        _instance_file.close()
        _instance_file = prev


def log(msg: str, *, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    try:
        _get_file().write(line + "\n")
    except Exception:
        pass
    if _instance_file is not None:
        try:
            _instance_file.write(line + "\n")
        except Exception:
            pass


def info(msg: str) -> None:
    log(msg, level="INFO")


def warn(msg: str) -> None:
    log(msg, level="WARN")


def error(msg: str) -> None:
    log(msg, level="ERROR")
