"""Run HybridLoc localization for a single (issue, repo, base_commit) — useful for debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybridloc.pipeline.orchestrate import HybridLocPipeline
from swebench_eval.load_dataset import base_commit_date


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/swe_bench_verified.yaml", type=Path)
    ap.add_argument("--issue", required=True, help="issue text or path to file")
    ap.add_argument("--repo-path", required=True, type=Path)
    ap.add_argument("--base-commit", default=None)
    ap.add_argument("--cache", default="data/graphs/_adhoc.pkl", type=Path)
    args = ap.parse_args()

    issue_text = (
        Path(args.issue).read_text()
        if Path(args.issue).exists()
        else args.issue
    )

    pipeline = HybridLocPipeline(config_path=args.config)
    bcd = (
        base_commit_date(args.repo_path, args.base_commit)
        if args.base_commit
        else None
    )
    bundle = pipeline.build_index(
        repo_root=args.repo_path,
        base_commit_sha=args.base_commit,
        base_commit_date=bcd,
        cache_path=args.cache,
    )
    pr = pipeline.localize(
        issue=issue_text,
        bundle=bundle,
        repo_root=args.repo_path,
        instance_id="adhoc",
    )
    out = {
        "termination": pr.termination,
        "think_calls": pr.think_high_calls,
        "ranked": [
            {
                "function_key": v.function_key,
                "score": v.score,
                "confidence": v.confidence,
                "runs_appearing": v.runs_appearing,
                "lines": v.suspect_lines,
                "chain": v.causal_chain,
            }
            for v in pr.ranked[:10]
        ],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
