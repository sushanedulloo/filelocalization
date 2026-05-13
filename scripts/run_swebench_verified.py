"""Headline script: run HybridLoc v2 on SWE-bench Verified and emit the metrics table."""

from __future__ import annotations

import argparse
from pathlib import Path

from swebench_eval.report import write_report
from swebench_eval.runner import HybridLocRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/swe_bench_verified.yaml", type=Path)
    ap.add_argument("--repos-root", default="data/repos", type=Path)
    ap.add_argument("--cache-root", default="data/graphs", type=Path)
    ap.add_argument("--out", default="results/swebench_verified.csv", type=Path)
    ap.add_argument("--out-md", default="results/swebench_verified.md", type=Path)
    ap.add_argument("--limit", default=None, type=int,
                    help="for smoke runs; default = full 500")
    ap.add_argument("--repos", default=None, nargs="*",
                    help="restrict to a subset of repos, e.g. django/django")
    args = ap.parse_args()

    runner = HybridLocRunner(
        config_path=args.config,
        repos_root=args.repos_root,
        cache_root=args.cache_root,
    )
    only = set(args.repos) if args.repos else None
    preds, golds = runner.run(limit=args.limit, only_repos=only)
    summary = write_report(preds, golds, out_csv=args.out, out_md=args.out_md)
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
