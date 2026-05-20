"""Build concept-cluster graphs for every unique (repo, base_commit) in
SWE-bench Verified, then exit. Does NOT run Stage 1-5 evaluation.

Use this to populate `data/graphs/` overnight. Subsequent evaluation runs
will reuse the cached graphs and be ~10-100x faster.

Resumable: skips any (repo, commit) already cached. Safe to interrupt
and resume.

Usage (inside tmux on the server):
    python scripts/build_all_graphs.py
    # or restrict:
    python scripts/build_all_graphs.py --repos django/django sympy/sympy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from hybridloc.log import info, warn
from hybridloc.pipeline.orchestrate import HybridLocPipeline
from swebench_eval.load_dataset import base_commit_date, clone_at, load_verified


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/swe_bench_verified.yaml", type=Path)
    ap.add_argument("--repos-root", default="data/repos", type=Path)
    ap.add_argument("--cache-root", default="data/graphs", type=Path)
    ap.add_argument("--repos", nargs="*", default=None,
                    help="restrict to a subset of repos, e.g. django/django")
    args = ap.parse_args()

    args.cache_root.mkdir(parents=True, exist_ok=True)
    args.repos_root.mkdir(parents=True, exist_ok=True)

    pipeline = HybridLocPipeline(config_path=args.config)
    ds = load_verified()

    # Build the set of unique (repo, base_commit) pairs
    only = set(args.repos) if args.repos else None
    seen: set[tuple[str, str]] = set()
    work: list[tuple[str, str]] = []
    for inst in ds:
        if only and inst["repo"] not in only:
            continue
        key = (inst["repo"], inst["base_commit"])
        if key in seen:
            continue
        seen.add(key)
        work.append(key)

    info(f"[build-all] {len(work)} unique (repo, commit) pairs to build")

    start = time.time()
    n_skipped = 0
    n_built = 0
    n_failed = 0

    bar = tqdm(work, desc="building graphs", unit="graph", dynamic_ncols=True)
    for repo, commit in bar:
        safe = repo.replace("/", "__")
        cache_path = args.cache_root / f"{safe}__{commit[:12]}.pkl"

        if cache_path.exists():
            n_skipped += 1
            bar.set_postfix(skipped=n_skipped, built=n_built, failed=n_failed)
            continue

        try:
            info(f"[build-all] {repo} @ {commit[:12]} ...")
            repo_path = clone_at(repo, commit, repos_root=args.repos_root)
            bcd = base_commit_date(repo_path, commit)
            pipeline.build_index(
                repo_root=repo_path,
                base_commit_sha=commit,
                base_commit_date=bcd,
                cache_path=cache_path,
            )
            n_built += 1
            info(f"[build-all] ✓ cached at {cache_path}")
        except Exception as e:
            n_failed += 1
            warn(f"[build-all] FAILED {repo} @ {commit[:12]}: {type(e).__name__}: {e}")
        bar.set_postfix(skipped=n_skipped, built=n_built, failed=n_failed)

    elapsed_min = (time.time() - start) / 60
    info(f"[build-all] done in {elapsed_min:.1f} min")
    info(f"[build-all]   built:   {n_built}")
    info(f"[build-all]   skipped: {n_skipped}")
    info(f"[build-all]   failed:  {n_failed}")


if __name__ == "__main__":
    main()
