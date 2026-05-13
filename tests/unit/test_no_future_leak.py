"""Critical bug-prevention test: RepoMem must never emit commits at or after base_commit_date.

This is the single most likely silent bug in the system: if it leaks, our SWE-bench
numbers are inflated by future information about the very fix we're supposed to find.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from hybridloc.graph.memory import collect_commits


def _make_fake_commit(sha: str, ts: datetime, files: list[str]):
    c = MagicMock()
    c.hexsha = sha
    c.message = f"msg {sha}"
    c.authored_date = int(ts.timestamp())
    c.stats.files = {f: {} for f in files}
    return c


def test_no_commits_at_or_after_base_commit_date(tmp_path):
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    fake_commits = [
        _make_fake_commit("c1_future", base + timedelta(days=2), ["a.py"]),
        _make_fake_commit("c2_at_base", base, ["b.py"]),
        _make_fake_commit("c3_past", base - timedelta(days=1), ["c.py"]),
        _make_fake_commit("c4_far_past", base - timedelta(days=400), ["d.py"]),
    ]

    fake_repo = MagicMock()
    fake_repo.iter_commits.return_value = iter(fake_commits)

    with patch("hybridloc.graph.memory.Repo", return_value=fake_repo):
        out = collect_commits(
            tmp_path,
            base_commit_sha="HEAD",
            base_commit_date=base,
            max_commits=100,
        )

    shas = [c.sha for c in out]
    assert "c1_future" not in shas, "future commit leaked"
    assert "c2_at_base" not in shas, "base commit itself leaked (must be strictly less)"
    assert "c3_past" in shas
    assert "c4_far_past" in shas


def test_max_commits_cap(tmp_path):
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    fake_commits = [
        _make_fake_commit(f"sha_{i}", base - timedelta(days=i + 1), ["x.py"])
        for i in range(50)
    ]
    fake_repo = MagicMock()
    fake_repo.iter_commits.return_value = iter(fake_commits)
    with patch("hybridloc.graph.memory.Repo", return_value=fake_repo):
        out = collect_commits(tmp_path, base_commit_date=base, max_commits=10)
    assert len(out) == 10
