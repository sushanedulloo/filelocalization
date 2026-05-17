"""Headline script: run HybridLoc v2 on SWE-bench Verified and emit the metrics table."""

from __future__ import annotations

import argparse
from pathlib import Path

from unidiff import PatchSet

from swebench_eval.report import write_report
from swebench_eval.runner import HybridLocRunner


def _patch_py_files(patch_text: str) -> int:
    """Count number of distinct .py files touched by a patch."""
    try:
        return len({pf.path for pf in PatchSet(patch_text) if pf.path.endswith(".py")})
    except Exception:
        return 0


def _patch_py_functions(patch_text: str) -> int:
    """Rough count of distinct hunks (proxy for number of changed functions)."""
    try:
        return sum(len(list(pf)) for pf in PatchSet(patch_text) if pf.path.endswith(".py"))
    except Exception:
        return 0


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
    ap.add_argument("--multi-file", action="store_true",
                    help="only instances where patch touches >1 .py file")
    ap.add_argument("--multi-func", action="store_true",
                    help="only instances where patch has >1 hunk (multiple functions changed)")
    args = ap.parse_args()

    runner = HybridLocRunner(
        config_path=args.config,
        repos_root=args.repos_root,
        cache_root=args.cache_root,
    )
    only = set(args.repos) if args.repos else None

    # pre-filter dataset for hard instance modes
    extra_filters = []
    if args.multi_file:
        extra_filters.append(lambda i: _patch_py_files(i["patch"]) > 1)
    if args.multi_func:
        extra_filters.append(lambda i: _patch_py_functions(i["patch"]) > 1)

    preds, golds = runner.run(limit=args.limit, only_repos=only, extra_filters=extra_filters)
    summary = write_report(preds, golds, out_csv=args.out, out_md=args.out_md)
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
