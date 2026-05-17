"""Restore cached graphs from HuggingFace Hub.

Run this once on a fresh machine (Kaggle, your teammate's laptop, a new
server) to skip the multi-hour concept-extraction step. The pipeline auto-
detects cached graphs at `data/graphs/<repo>__<commit>.pkl` and reuses them.

Usage:
    python scripts/restore_cache.py

Optional env vars:
    HF_REPO_ID         repo id, e.g. "yourname/hybridloc-cache"
    INCLUDE_NIM_CACHE  set to "1" to also restore data/nim_cache/
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

    repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id:
        repo_id = input("Enter HF repo id (e.g. yourname/hybridloc-cache): ").strip()
        if not repo_id:
            sys.exit("No repo id provided")

    allow = ["graphs/*"]
    if os.environ.get("INCLUDE_NIM_CACHE") == "1":
        allow.append("nim_cache/*")

    print(f"Downloading from {repo_id} ...")
    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir="data/_hf_cache_tmp",
        allow_patterns=allow,
    )
    local_path = Path(local)

    # Move into expected locations
    graphs_src = local_path / "graphs"
    if graphs_src.exists():
        Path("data/graphs").mkdir(parents=True, exist_ok=True)
        for f in graphs_src.iterdir():
            shutil.move(str(f), f"data/graphs/{f.name}")
        print(f"  graphs restored to data/graphs/ ({len(list(Path('data/graphs').glob('*.pkl')))} files)")

    nim_src = local_path / "nim_cache"
    if nim_src.exists():
        Path("data/nim_cache").mkdir(parents=True, exist_ok=True)
        for f in nim_src.iterdir():
            shutil.move(str(f), f"data/nim_cache/{f.name}")
        print("  nim_cache restored to data/nim_cache/")

    # Clean up the staging dir
    shutil.rmtree("data/_hf_cache_tmp", ignore_errors=True)

    print("\nDone. The pipeline will now skip Stage 2 for cached repos.")


if __name__ == "__main__":
    main()
