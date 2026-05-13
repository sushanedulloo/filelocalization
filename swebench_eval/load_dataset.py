"""Wraps `datasets.load_dataset` for SWE-bench Verified with a cached repo cloner."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


def load_verified():
    return load_dataset("princeton-nlp/SWE-bench_Verified", split="test")


def clone_at(
    repo_full_name: str,
    base_commit: str,
    *,
    repos_root: Path,
) -> Path:
    """Clone (or reuse) the repo and check out base_commit. Returns the worktree path."""
    repos_root.mkdir(parents=True, exist_ok=True)
    safe = repo_full_name.replace("/", "__")
    base = repos_root / safe
    if not base.exists():
        url = f"https://github.com/{repo_full_name}.git"
        subprocess.run(
            ["git", "clone", "--quiet", url, str(base)],
            check=True,
        )
    # checkout the base commit (detached HEAD; idempotent)
    subprocess.run(
        ["git", "-C", str(base), "fetch", "--quiet", "origin", base_commit],
        check=False,
    )
    subprocess.run(
        ["git", "-C", str(base), "checkout", "--quiet", base_commit],
        check=True,
    )
    return base


def base_commit_date(repo_path: Path, base_commit: str) -> datetime:
    """ISO date of the base commit, used by RepoMem to prevent future-leak."""
    out = subprocess.run(
        ["git", "-C", str(repo_path), "show", "-s", "--format=%cI", base_commit],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return datetime.fromisoformat(out).astimezone(timezone.utc)
