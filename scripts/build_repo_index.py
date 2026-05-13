"""Build/cache the skeleton index for a repository.

Usage:
    python scripts/build_repo_index.py --repo-path /path/to/repo --out data/skeletons/<name>.jsonl
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hybridloc.parsing.skeleton import build_repo_skeleton, save_skeletons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-path", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", default=8, type=int)
    args = ap.parse_args()

    t0 = time.perf_counter()
    skels = build_repo_skeleton(args.repo_path, max_workers=args.workers)
    save_skeletons(skels, args.out)
    print(
        f"wrote {len(skels)} skeletons to {args.out} "
        f"in {time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
