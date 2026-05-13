"""RepoMem: mine commit history for EVOLVED_BY and CO_EVOLVED edges.

CRITICAL: must respect `base_commit_date` to prevent future-leak in eval.
The unit test `test_no_future_leak` enforces this.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
from git import Repo

from .nodes import EdgeType, NodeData, NodeType, fid_commit, fid_file


@dataclass
class CommitInfo:
    sha: str
    message: str
    authored_at: datetime
    files: list[str] = field(default_factory=list)


def collect_commits(
    repo_path: Path,
    *,
    base_commit_sha: str | None = None,
    base_commit_date: datetime | None = None,
    max_commits: int = 7000,
) -> list[CommitInfo]:
    """Walk git history from `base_commit_sha` (HEAD if None) backward.

    Returns commits with `authored_at < base_commit_date` (strictly less),
    most recent first, capped at `max_commits`.
    """
    repo = Repo(repo_path)
    rev = base_commit_sha or "HEAD"
    out: list[CommitInfo] = []
    for c in repo.iter_commits(rev, max_count=max_commits * 2):
        ts = datetime.fromtimestamp(c.authored_date, tz=timezone.utc)
        if base_commit_date is not None and ts >= base_commit_date:
            # this also excludes the base commit itself
            continue
        try:
            files = list(c.stats.files.keys())
        except Exception:
            files = []
        out.append(
            CommitInfo(
                sha=c.hexsha,
                message=(c.message if isinstance(c.message, str) else c.message.decode("utf-8", errors="replace")),
                authored_at=ts,
                files=files,
            )
        )
        if len(out) >= max_commits:
            break
    return out


def add_memory_edges(
    g: nx.MultiDiGraph,
    commits: list[CommitInfo],
    *,
    max_files: int = 200,
    co_evolved_min_count: int = 3,
) -> None:
    """Adds COMMIT nodes, EVOLVED_BY edges, CO_EVOLVED edges to the graph."""
    if not commits:
        return

    # how often each file is touched
    file_counts: Counter[str] = Counter()
    for c in commits:
        for f in c.files:
            file_counts[f] += 1

    # cap to top-N most-edited files (RepoMem semantic-memory layer)
    keep_files = {f for f, _ in file_counts.most_common(max_files)}

    # commit nodes + EVOLVED_BY
    for c in commits:
        cid = fid_commit(c.sha)
        g.add_node(
            cid,
            data=NodeData(
                node_type=NodeType.COMMIT,
                name=c.sha[:12],
                extra={
                    "message": c.message[:500],
                    "authored_at": c.authored_at.isoformat(),
                },
            ),
        )
        for f in c.files:
            if f not in keep_files:
                continue
            file_id = fid_file(f)
            if file_id in g:
                g.add_edge(
                    file_id,
                    cid,
                    key=EdgeType.EVOLVED_BY.value,
                    edge_type=EdgeType.EVOLVED_BY,
                )

    # CO_EVOLVED: count file pairs co-edited in the same commit
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for c in commits:
        files_in_repo = sorted({f for f in c.files if f in keep_files})
        for i in range(len(files_in_repo)):
            for j in range(i + 1, len(files_in_repo)):
                pair_counts[(files_in_repo[i], files_in_repo[j])] += 1

    for (a, b), n in pair_counts.items():
        if n < co_evolved_min_count:
            continue
        ai, bi = fid_file(a), fid_file(b)
        if ai in g and bi in g:
            g.add_edge(
                ai,
                bi,
                key=EdgeType.CO_EVOLVED.value,
                edge_type=EdgeType.CO_EVOLVED,
                count=n,
            )
            g.add_edge(
                bi,
                ai,
                key=EdgeType.CO_EVOLVED.value,
                edge_type=EdgeType.CO_EVOLVED,
                count=n,
            )
